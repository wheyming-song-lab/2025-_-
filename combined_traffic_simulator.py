import pandas as pd
import joblib
import numpy as np
from tqdm import tqdm
from holidays import holiday_dates
from scipy.stats import truncnorm
from sklearn.mixture import GaussianMixture

class CombinedTrafficSimulator:
    def __init__(self, excel_files, reg_model_paths, vol_model_paths,
                 direction="all", flow_limit_csvs=None, highway_lengths_km=None):
        self.directions = {
            "南下": ["0021", "0023"],
            "北上": ["0022", "0024"]
        }

        if direction == "all":
            self.selected_keys = self.directions["南下"] + self.directions["北上"]
        elif direction in self.directions:
            self.selected_keys = self.directions[direction]
        else:
            raise ValueError("direction 必須為 '南下'、'北上' 或 'all'")

        self.reg_models = {}
        self.vol_models = {}
        self.flow_limit = {}
        self.vehicles = []

        self.highway_lengths = {k: int(highway_lengths_km.get(k, 1.9) * 1000)
                                for k in self.selected_keys}

        if flow_limit_csvs:
            for key in self.selected_keys:
                if key in flow_limit_csvs:
                    limit_df = pd.read_csv(flow_limit_csvs[key])
                    self.flow_limit[key] = dict(zip(limit_df["時間字串"], limit_df["流量上限"]))
            print(f"✅ 已載入 flow limit（共 {len(self.flow_limit)} 路段）")

        df_list = []
        for key in self.selected_keys:
            df = pd.read_excel(excel_files[key])
            df["路段"] = key
            df["方向"] = "南下" if key in self.directions["南下"] else "北上"
            df["日期"] = pd.to_datetime(df["時間點"]).dt.strftime("%Y-%m-%d")
            df_list.append(df)

        self.df = pd.concat(df_list, ignore_index=True)
        self.df["時間點"] = pd.to_datetime(self.df["時間點"])
        # ✅ 每個時間字串（HH:MM）對應的歷史標準差
        self.df["時間字串"] = self.df["時間點"].dt.strftime("%H:%M")
        self.df_all_raw = self.df.copy()  # 原始有 TravelSpeed 的版本
        self.sigma_by_time = (
            self.df.groupby("時間字串")["TravelSpeed"]
            .std()
            .fillna(3)
            .to_dict()
        )
        print(f"✅ 共讀入 {len(self.df)} 筆資料")

    def predict_and_simulate(self, probabilistic=False):
        from scipy.stats import truncnorm
        from scipy.signal import find_peaks

        vehicle_id = 0
        records = []
        self.vehicles = []

        df_sorted = self.df.sort_values(["時間點", "路段"])
        entry_only_keys = [k for k in self.selected_keys if k in ["0021", "0022"]]

        for key in tqdm(entry_only_keys, desc="僅從進入路段模擬產車"):
            sub_df = df_sorted[df_sorted["路段"] == key].copy()
            sub_df = sub_df.rename(columns={
                "預測車速": "base_speed",
                "預測車流量": "flow"
            })

            for _, row in tqdm(sub_df.iterrows(), total=len(sub_df), desc=f"  處理路段 {key}"):
                time = row["時間點"]
                direction = row["方向"]
                time_str = time.strftime("%H:%M")
                flow = row["flow"]
                base_speed = row["base_speed"]

                if self.flow_limit.get(key) and time_str in self.flow_limit[key]:
                    flow = min(flow, self.flow_limit[key][time_str])

                if flow < 0 or np.isnan(flow):
                    continue

                num_vehicles = int(round(flow))

                if probabilistic:
                    # 🔍 擬合真實車速分布的 GMM
                    real_data = self.df_all_raw[
                        (self.df_all_raw["路段"] == key) &
                        (self.df_all_raw["時間字串"] == time_str)
                        ]["TravelSpeed"].dropna()

                    if len(real_data) >= 50:
                        counts, bins = np.histogram(real_data, bins=30)
                        prominence_thresh = max(np.max(counts) * 0.1, 100)
                        height_thresh = max(np.max(counts) * 0.15, 80)
                        peaks, _ = find_peaks(
                            counts,
                            prominence=prominence_thresh,
                            height=height_thresh,
                            distance=2
                        )

                        num_components = max(1, min(len(peaks), 4))  # 至少1群，最多4群

                        gmm = GaussianMixture(n_components=num_components, random_state=42)
                        gmm.fit(real_data.values.reshape(-1, 1))
                        weights = gmm.weights_
                        mus = gmm.means_.flatten()
                        sigmas = np.sqrt(gmm.covariances_).flatten()

                        vehicle_speeds = []
                        for _ in range(num_vehicles):
                            comp = np.random.choice(np.arange(num_components), p=weights)
                            mu = mus[comp]
                            sigma = sigmas[comp]
                            a, b = (mu - 4 * sigma - mu) / sigma, (mu + 4 * sigma - mu) / sigma
                            speed = truncnorm.rvs(a, b, loc=mu, scale=sigma)
                            vehicle_speeds.append(speed)
                    else:
                        # fallback：單一截尾常態
                        sigma = self.sigma_by_time.get(time_str, 3)
                        a, b = (base_speed - 4 * sigma - base_speed) / sigma, (
                                    base_speed + 4 * sigma - base_speed) / sigma
                        vehicle_speeds = truncnorm.rvs(a, b, loc=base_speed, scale=sigma, size=num_vehicles)

                    for i in range(num_vehicles):
                        speed = vehicle_speeds[i]
                        speed_mps = speed / 3.6
                        dist = min(speed_mps * 300, self.highway_lengths[key])
                        offset = np.random.exponential(scale=1.5)
                        enter_time = time + pd.to_timedelta(offset, unit='s')

                        records.append({
                            "車輛ID": vehicle_id,
                            "進入時間": enter_time,
                            "方向": direction,
                            "路段": key,
                            "預測車速": speed,
                            "模擬行駛距離": dist
                        })
                        vehicle_id += 1

                else:
                    # 非機率模擬
                    num_vehicles = min(num_vehicles, 1000)
                    speed_mps = base_speed / 3.6
                    dist = min(speed_mps * 300, self.highway_lengths[key])

                    for _ in range(num_vehicles):
                        records.append({
                            "車輛ID": vehicle_id,
                            "進入時間": time,
                            "方向": direction,
                            "路段": key,
                            "預測車速": base_speed,
                            "模擬行駛距離": dist
                        })
                        vehicle_id += 1

        self.vehicles = pd.DataFrame(records)
        print(f"\n✅ 模擬完成，總車輛數：{len(self.vehicles):,}")
        print("\n📊 各路段模擬車輛數：")
        print(self.vehicles["路段"].value_counts())

        total_count = len(self.vehicles)
        avg_speed = self.vehicles["預測車速"].mean()
        min_time = self.vehicles["進入時間"].min().strftime("%H:%M:%S")
        max_time = self.vehicles["進入時間"].max().strftime("%H:%M:%S")
        unique_roads = sorted(self.vehicles["路段"].unique())

        print("\n📋 模擬總結：")
        print(f"🔢 總車輛數：{total_count:,} 台")
        print(f"🚗 平均預測車速：{avg_speed:.2f} km/h")
        print(f"🕒 模擬時段：{min_time} ～ {max_time}")
        print(f"📍 涉及路段：{len(unique_roads)} 段（{', '.join(unique_roads)}）")

    def get_result_df(self):
        if self.vehicles is not None:
            return self.vehicles.copy()
        else:
            print("⚠️ 尚未模擬，請先執行 predict_and_simulate()")
            return None

    def export_vehicle_time_info(self, output_csv="vehicle_passage_time.csv"):
        if self.vehicles is None or self.vehicles.empty:
            print("⚠️ 尚未模擬，請先執行 predict_and_simulate()")
            return

        df = self.vehicles.copy()
        df = df.sort_values(["車輛ID", "進入時間"])
        group_cols = ["車輛ID", "方向"]

        records = []
        grouped = df.groupby(group_cols)

        for (vid, direction), group in tqdm(grouped, desc="⏱️ 匯出進出與離段時間", total=len(grouped)):
            enter_time = group["進入時間"].min()
            avg_speed = group["預測車速"].mean()
            speed_mps = avg_speed / 3.6

            if direction == "南下":
                dist1 = self.highway_lengths.get("0021", 0)
                dist2 = self.highway_lengths.get("0023", 0)
            else:
                dist1 = self.highway_lengths.get("0022", 0)
                dist2 = self.highway_lengths.get("0024", 0)

            total_dist = dist1 + dist2
            duration_total = total_dist / speed_mps
            duration_1 = dist1 / speed_mps
            duration_2 = dist2 / speed_mps

            exit_time = enter_time + pd.to_timedelta(duration_total, unit="s")
            exit_road1 = enter_time + pd.to_timedelta(duration_1, unit="s")
            exit_road2 = exit_time

            records.append({
                "車輛ID": vid,
                "方向": direction,
                "進入時間": enter_time,
                "離開時間": exit_time,
                "離開時間_路段1": exit_road1,
                "離開時間_路段2": exit_road2,
                "平均車速(km/h)": round(avg_speed, 2),
                "總距離(m)": total_dist,
                "通過時間(sec)": round(duration_total, 1)
            })

        df_out = pd.DataFrame(records)
        df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"✅ 已輸出每車進出與路段離開時間資訊到：{output_csv}")

# ✅ 新增函數：建立平日平均輸入資料
def build_weekday_average_df(df_all, holiday_dates):
    df_all = df_all.copy()
    df_all["時間點"] = pd.to_datetime(df_all["時間點"])
    df_all["日期"] = df_all["時間點"].dt.strftime("%Y-%m-%d")
    df_all["時間字串"] = df_all["時間點"].dt.strftime("%H:%M")
    df_all["Weekday"] = df_all["時間點"].dt.weekday

    df_all = df_all[
        (df_all["Weekday"] < 5) &
        (~df_all["日期"].isin(holiday_dates))
    ]

    df_avg = df_all.groupby(["路段", "時間字串", "方向"]).agg({
        "TravelSpeed": "mean",
        "總車輛數": "mean"
    }).reset_index()

    df_avg["時間點"] = pd.to_datetime("2023-10-25 " + df_avg["時間字串"])
    df_avg = df_avg.rename(columns={"TravelSpeed": "預測車速", "總車輛數": "預測車流量"})

    return df_avg
