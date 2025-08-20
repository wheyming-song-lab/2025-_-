import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import truncnorm
from scipy.signal import find_peaks
from datetime import timedelta
from holidays import holiday_dates
from tqdm import tqdm

# ====== 參數 ======
np.random.seed(42)
MIN_SPEED, MAX_SPEED = 20, 120
ROAD_LENGTH = 2000            # 0024 單一路段長度 (m)；可依實際調整
MAX_COMPONENTS = 6
AUTO_COMPONENTS = True
FIXED_COMPONENTS = 2
GMM_MIN_SAMPLES = 40          # 同一時段(HH:MM)樣本數至少要這麼多才擬合 GMM
BIN_SECONDS = 300             # 每 5 分鐘窗長度
BASE_DATE = "2025-01-01"      # 24h 輸出使用的基準日期（僅作標籤）
P_RAMP = 0.25                  # 匝道比例

FILE_0024 = r"C:\Users\User\Desktop\交通部\交通部競賽\data_2025\0024_三重交流道_到_台北交流道_final.xlsx"

# ====== 過濾：剔除節日與週末 ======
def filter_weekdays(df):
    df = df.copy()
    df_days = df["時間點"].dt.floor("D")
    df = df[~df_days.isin(pd.to_datetime(holiday_dates))]
    df = df[df["時間點"].dt.weekday < 5]
    df["時間段"] = df["時間點"].dt.floor("5min")
    df["時間_hm"] = df["時間段"].dt.strftime("%H:%M")
    return df

# ====== 生成 24h × 5min 平均輪廓 ======
def build_24h_profile(df):
    # 對所有平日資料，按 HH:MM 聚合
    prof = df.groupby("時間_hm").agg(
        avg_speed=("TravelSpeed", "mean"),
        avg_flow=("總車輛數", "mean"),
        cnt=("TravelSpeed", "count")
    ).reset_index()

    # 確保 00:00～23:55 的 288 格都齊全（缺的補 0/NaN 再處理）
    full_hm = pd.date_range(BASE_DATE, periods=288, freq="5min").strftime("%H:%M")
    prof = prof.set_index("時間_hm").reindex(full_hm).reset_index().rename(columns={"index": "時間_hm"})

    # 缺值處理
    med_v = prof["avg_speed"].median() if prof["avg_speed"].notna().sum() > 0 else 80.0
    prof["avg_speed"] = prof["avg_speed"].fillna(med_v)
    prof["avg_flow"] = prof["avg_flow"].fillna(0.0).clip(lower=0.0)

    # 加上基準日期的實際時間軸
    prof["時間段"] = pd.to_datetime(BASE_DATE + " " + prof["時間_hm"])
    return prof[["時間段", "時間_hm", "avg_speed", "avg_flow"]]

# ====== 指數分布到達時刻（已知目標到達數）======
def exponential_offsets(count, T=BIN_SECONDS):
    if count <= 0:
        return np.array([])
    lam = count / T
    if lam <= 0:
        return np.array([])
    intervals = np.random.exponential(scale=1.0/lam, size=count)
    offsets = np.cumsum(intervals)
    offsets[offsets > T] = T
    return np.sort(offsets)

# ====== GMM 擬合（以全年平日、同一 HH:MM 样本）======
gmm_cache = {}

def fit_gmm_for_hm(hm, samples_1d):
    if hm in gmm_cache:
        return gmm_cache[hm]
    x = samples_1d.reshape(-1, 1)
    if AUTO_COMPONENTS:
        counts, _ = np.histogram(samples_1d, bins=30)
        prom = max(np.max(counts) * 0.08, 20)
        peaks, _ = find_peaks(counts, prominence=prom, distance=2)
        n_components = int(np.clip(len(peaks), 1, MAX_COMPONENTS))
    else:
        n_components = FIXED_COMPONENTS
    init_means = np.linspace(samples_1d.min(), samples_1d.max(), n_components).reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=n_components,
        means_init=init_means,
        covariance_type='full',
        reg_covar=1e-3,
        random_state=42
    ).fit(x)
    gmm_cache[hm] = gmm
    return gmm

def sample_gmm_trunc(gmm, n, vmin=MIN_SPEED, vmax=MAX_SPEED):
    samples = []
    stds = np.sqrt(gmm.covariances_.reshape(-1))
    stds = np.clip(stds, 1.0, None)
    counts = np.random.multinomial(n, gmm.weights_)
    for cnt, mu, sigma in zip(counts, gmm.means_.flatten(), stds):
        if cnt > 0:
            a, b = (vmin - mu) / sigma, (vmax - mu) / sigma
            samples.extend(truncnorm.rvs(a, b, loc=mu, scale=sigma, size=cnt))
    return np.array(samples)

def sample_speeds(hm, count, hm_pool, fallback_mean):
    if count <= 0:
        return np.array([])
    arr = hm_pool.get(hm, np.array([]))
    if arr.size >= GMM_MIN_SAMPLES:
        gmm = fit_gmm_for_hm(hm, arr)
        vals = sample_gmm_trunc(gmm, count)
    else:
        sigma = 6.0
        a, b = (MIN_SPEED - fallback_mean) / sigma, (MAX_SPEED - fallback_mean) / sigma
        vals = truncnorm.rvs(a, b, loc=fallback_mean, scale=sigma, size=count)
    return np.clip(vals, MIN_SPEED, MAX_SPEED)

# ====== 主程式 ======
if __name__ == "__main__":
    # 1) 讀檔 + 過濾
    df = pd.read_excel(FILE_0024, parse_dates=["時間點"])
    df = filter_weekdays(df)

    # 2) 建 24h 平均輪廓
    profile = build_24h_profile(df)

    # 3) 構建全年平日同一 HH:MM 車速樣本池
    hm_pool = (
        df.groupby("時間_hm")["TravelSpeed"]
          .apply(lambda s: s.dropna().astype(float).values)
          .to_dict()
    )

    # 4) 依輪廓生成單日 24h 車輛資料（加入來源）
    records = []
    vehicle_id = 1
    for _, row in tqdm(profile.iterrows(), total=len(profile), desc="🚗 生成 24h (0024)"):
        hm = row["時間_hm"]
        t_start = row["時間段"]
        mean_speed = float(row["avg_speed"])
        count = int(round(row["avg_flow"]))
        if count <= 0:
            continue
        speeds = sample_speeds(hm, count, hm_pool, mean_speed)
        offsets = exponential_offsets(count, BIN_SECONDS)
        enter_times = [t_start + timedelta(seconds=float(o)) for o in offsets]
        for v in range(count):
            v_kmh = float(speeds[v])
            travel_time = ROAD_LENGTH / (v_kmh / 3.6)
            exit_time = enter_times[v] + timedelta(seconds=travel_time)
            # 匝道或主線標記
            src = "匝道" if np.random.rand() < P_RAMP else "主線"
            records.append([
                vehicle_id, t_start, enter_times[v], v_kmh, travel_time, exit_time, src
            ])
            vehicle_id += 1

    # 5) 輸出
    cols = ["車輛ID", "時段", "進入時間", "期望速度(km/h)", "行駛時間(s)", "退出時間", "來源"]
    df_out = pd.DataFrame(records, columns=cols)
    out_path = r"C:\Users\User\Desktop\交通部\交通部競賽\new_train\xgboost\v5模擬\sim_0024_outputs\figs\vehicles_0024_24h_profile.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 6) 摘要
    print("====== 生成完成（僅 24 小時）======")
    print("總 5 分鐘區間數：", profile.shape[0])
    print("總車輛數：", df_out.shape[0])
    print("輸出檔案：", out_path)
