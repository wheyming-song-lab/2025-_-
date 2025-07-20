#goal：建立時速的直方圖與truncated normal
#未做：混合常態

import pandas as pd
from combined_traffic_simulator import CombinedTrafficSimulator, build_weekday_average_df
from holidays import holiday_dates
from flow_limit_generator import generate_flow_limit_table
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy.stats import norm
import numpy as np
from scipy.stats import truncnorm
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture

matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'  # 微軟正黑體
matplotlib.rcParams['axes.unicode_minus'] = False  # 負號正常顯示

# 1️⃣ 四段資料路徑
excel_files = {
    "0021": r"C:\Users\User\Desktop\父子關係\父子關係\品code\traffic_pipeline\output\0021_圓山交流道_到_台北交流道_final.xlsx",
    "0022": r"C:\Users\User\Desktop\父子關係\父子關係\品code\traffic_pipeline\output\0022_台北交流道_到_圓山交流道_final.xlsx",
    "0023": r"C:\Users\User\Desktop\父子關係\父子關係\品code\traffic_pipeline\output\0023_台北交流道_到_三重交流道_final.xlsx",
    "0024": r"C:\Users\User\Desktop\父子關係\父子關係\品code\traffic_pipeline\output\0024_三重交流道_到_台北交流道_final.xlsx"
}

# 2️⃣ 各段模型路徑
reg_model_paths = {
    "0021": r"C:\Users\User\Desktop\父子關係\父子關係\traffic\品\saved_models_2\0021_reg.pkl",
    "0022": r"C:\Users\User\Desktop\父子關係\父子關係\traffic\品\saved_models_2\0022_reg.pkl",
    "0023": r"C:\Users\User\Desktop\父子關係\父子關係\traffic\品\saved_models_2\0023_reg.pkl",
    "0024": r"C:\Users\User\Desktop\父子關係\父子關係\traffic\品\saved_models_2\0024_reg.pkl"
}
vol_model_paths = {
    "0021": r"C:\Users\User\Desktop\父子關係\父子關係\traffic\品\saved_models_2\0021_vol.pkl",
    "0022": r"C:\Users\User\Desktop\父子關係\父子關係\traffic\品\saved_models_2\0022_vol.pkl",
    "0023": r"C:\Users\User\Desktop\父子關係\父子關係\traffic\品\saved_models_2\0023_vol.pkl",
    "0024": r"C:\Users\User\Desktop\父子關係\父子關係\traffic\品\saved_models_2\0024_vol.pkl"
}

# 3️⃣ 路段長度（單位：公里）
highway_lengths_km = {
    "0021": 1.9,
    "0022": 1.9,
    "0023": 2.0,
    "0024": 2.0
}

# 4️⃣ 產生每段路的 flow limit 表（如已存在可略過）
for key in excel_files:
    generate_flow_limit_table(
        excel_file=excel_files[key],
        output_csv=f"flow_limit_{key}.csv"
    )

# 5️⃣ flow limit 對應表
flow_limit_csvs = {
    "0021": "flow_limit_0021.csv",
    "0022": "flow_limit_0022.csv",
    "0023": "flow_limit_0023.csv",
    "0024": "flow_limit_0024.csv"
}

# 2️⃣ 合併所有資料
df_all = []
for key in excel_files:
    df = pd.read_excel(excel_files[key])
    df["路段"] = key
    df["方向"] = "南下" if key in ["0021", "0023"] else "北上"
    df_all.append(df)
df_all = pd.concat(df_all, ignore_index=True)

# 3️⃣ 建立平日平均模擬資料
df_avg = build_weekday_average_df(df_all, holiday_dates)

# 4️⃣ 初始化模擬器並餵入平均資料
simulator = CombinedTrafficSimulator(
    excel_files=excel_files,
    reg_model_paths=reg_model_paths,
    vol_model_paths=vol_model_paths,
    direction="all",
    flow_limit_csvs=flow_limit_csvs,
    highway_lengths_km=highway_lengths_km
)
simulator.df = df_avg.copy()

# 5️⃣ 執行模擬
simulator.predict_and_simulate(probabilistic=True)
df_sim = simulator.get_result_df()


def plot_hourly_speed_count_distribution(df_sim, direction="南下"):
    df = df_sim[df_sim["方向"] == direction].copy()
    df["進入時間"] = pd.to_datetime(df["進入時間"])
    df["hour"] = df["進入時間"].dt.hour

    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(22, 12))
    axes = axes.flatten()

    hourly_avg = df.groupby("hour")["預測車速"].mean()

    print(f"📊 繪製 {direction} 方向車速直方圖（含統計值與常態擬合）")
    for hour in tqdm(range(24), desc=f"{direction} 繪圖中"):
        ax = axes[hour]
        hour_data = df[df["hour"] == hour]["預測車速"]

        if hour_data.empty:
            ax.set_visible(False)
            continue

        # 畫直方圖（Y 軸為車輛數）
        sns.histplot(hour_data, kde=False, bins=30, stat="count",
                     ax=ax, color='orange' if direction == "南下" else 'blue',
                     edgecolor='black', alpha=0.85)

        ax.set_xlim(0, 120)
        ax.set_ylim(0, None)
        ax.set_title(f"{hour:02d}:00 - {hour+1:02d}:00", fontsize=10)
        ax.set_xlabel("車速 (km/h)", fontsize=8)
        ax.set_ylabel("車輛數", fontsize=8)
        ax.tick_params(axis='both', labelsize=8)

        # 統計值（μ, σ, N）
        mu = hour_data.mean()
        std = hour_data.std()
        count = len(hour_data)
        ax.text(0.02, 0.90,
                f"μ={mu:.1f}, σ={std:.1f}, N={count}",
                transform=ax.transAxes,
                fontsize=8, color='black',
                verticalalignment='top', horizontalalignment='left')

        # 平均車速紅線
        if hour in hourly_avg:
            ax.axvline(hourly_avg.loc[hour], color='red', linestyle='--', linewidth=1)

        # 常態分布擬合線
        try:
            fit_mu, fit_std = norm.fit(hour_data)
            x = np.linspace(0, 120, 200)
            bin_width = (hour_data.max() - hour_data.min()) / 30
            y = norm.pdf(x, fit_mu, fit_std) * count * bin_width
            ax.plot(x, y, 'black', linestyle='-', linewidth=1.2)
        except Exception as e:
            print(f"⚠️ 擬合失敗（hour={hour}）: {e}")

    fig.suptitle(f"{direction}方向 每小時車速分布圖（含統計 + 常態擬合）", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"speed_count_distribution_{direction}.png", dpi=300)
    plt.show()


print("畫圖中")
#plot_hourly_speed_count_distribution(df_sim, direction="南下")
#plot_hourly_speed_count_distribution(df_sim, direction="北上")

# 繪製真實資料直方圖
def plot_combined_real_and_simulated_distribution(df_real, df_sim, sigma_by_time, direction="南下"):
    color = '#FFA500' if direction == "南下" else '#1E90FF'
    df_real = df_real[df_real["方向"] == direction].copy()
    df_sim = df_sim[df_sim["方向"] == direction].copy()

    df_real["時間點"] = pd.to_datetime(df_real["時間點"])
    df_sim["進入時間"] = pd.to_datetime(df_sim["進入時間"])
    df_real["hour"] = df_real["時間點"].dt.hour
    df_sim["hour"] = df_sim["進入時間"].dt.hour

    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(22, 12))
    axes = axes.flatten()

    for hour in range(24):
        ax = axes[hour]

        real_data = df_real[df_real["hour"] == hour]["TravelSpeed"]
        sim_data = df_sim[df_sim["hour"] == hour]["預測車速"]

        if real_data.empty or sim_data.empty:
            ax.set_visible(False)
            continue

        # 🚨 多峰偵測（動態門檻）
        counts, bins = np.histogram(real_data, bins=30)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        count_max = np.max(counts)
        prominence_thresh = max(count_max * 0.1, 100)
        height_thresh = max(count_max * 0.15, 80)

        from scipy.signal import find_peaks
        peaks, properties = find_peaks(
            counts,
            prominence=prominence_thresh,
            height=height_thresh,
            distance=2
        )
        peak_positions = bin_centers[peaks]
        num_peaks = len(peaks)
        is_multi_peak = num_peaks >= 2 and (peak_positions[-1] - peak_positions[0]) > 10

        # 🔶 真實資料直方圖
        sns.histplot(real_data, kde=False, bins=8, stat="count",
                     ax=ax, color=color, edgecolor='black', alpha=0.7, label="真實")

        # 🔷 模擬分布線：根據多峰自動調整 GMM 成分數
        sim_data_nonan = sim_data.dropna().values.reshape(-1, 1)
        if len(sim_data_nonan) >= 50:
            n_components = min(max(num_peaks, 1), 4)  # 至少 1 群，最多 4 群
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(sim_data_nonan)
            weights = gmm.weights_
            mus = gmm.means_.flatten()
            sigmas = np.sqrt(gmm.covariances_).flatten()

            x = np.linspace(0, 120, 300)
            bin_width = (real_data.max() - real_data.min()) / 30
            total_pdf = np.zeros_like(x)

            for w, mu, sigma in zip(weights, mus, sigmas):
                a, b = (mu - 4 * sigma - mu) / sigma, (mu + 4 * sigma - mu) / sigma
                pdf = truncnorm.pdf(x, a, b, loc=mu, scale=sigma)
                total_pdf += w * pdf

            scaled_pdf = total_pdf * len(real_data) * bin_width
            ax.plot(x, scaled_pdf, color='darkgreen', linewidth=1.5,
                    label=f"模擬混合截尾分布（{n_components} 群）")

            mu = np.sum(weights * mus)  # 混合平均 μ
            sigma = np.sqrt(np.sum(weights * (sigmas ** 2 + (mus - mu) ** 2)))  # 混合總體 σ
        else:
            # fallback 單一截尾常態
            mu = sim_data.mean()
            sigma = sigma_by_time.get(f"{hour:02d}:00", 3)
            a, b = (mu - 4 * sigma - mu) / sigma, (mu + 4 * sigma - mu) / sigma
            x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
            x = np.clip(x, 0, 120)
            pdf = truncnorm.pdf(x, a, b, loc=mu, scale=sigma)
            bin_width = (real_data.max() - real_data.min()) / 30
            scaled_pdf = pdf * len(real_data) * bin_width
            ax.plot(x, scaled_pdf, color='darkgreen', linewidth=1.5, label="模擬截尾分布")

        # 圖表標題與格式
        ax.set_xlim(0, 120)
        title = f"{hour:02d}:00 - {hour + 1:02d}:00"
        if is_multi_peak:
            title += f"{num_peaks} 峰"
        ax.set_title(title, fontsize=10)

        ax.set_xlabel("車速 (km/h)", fontsize=8)
        ax.set_ylabel("車輛數", fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.text(0.02, 0.95, f"μ={mu:.1f}, σ={sigma:.1f}", transform=ax.transAxes, fontsize=8)

    fig.suptitle(f"{direction}方向 每小時車速分布（真實直方圖 + 截尾常態 PDF）", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



plot_combined_real_and_simulated_distribution(df_all, df_sim, simulator.sigma_by_time, direction="南下")
plot_combined_real_and_simulated_distribution(df_all, df_sim, simulator.sigma_by_time, direction="北上")

def plot_multi_peak_hours(df_real, df_sim, sigma_by_time, direction="南下"):
    color = '#FFA500' if direction == "南下" else '#1E90FF'
    df_real = df_real[df_real["方向"] == direction].copy()
    df_sim = df_sim[df_sim["方向"] == direction].copy()

    df_real["時間點"] = pd.to_datetime(df_real["時間點"])
    df_sim["進入時間"] = pd.to_datetime(df_sim["進入時間"])
    df_real["hour"] = df_real["時間點"].dt.hour
    df_sim["hour"] = df_sim["進入時間"].dt.hour

    multi_peak_hours = []

    for hour in range(24):
        real_data = df_real[df_real["hour"] == hour]["TravelSpeed"]
        if len(real_data) < 50:
            continue

        counts, bins = np.histogram(real_data, bins=30)
        peaks, _ = find_peaks(counts, prominence=200)  # 可調參數
        if len(peaks) >= 2:
            multi_peak_hours.append(hour)

    print(f"📌 偵測到多峰時段：{multi_peak_hours}")

    n = len(multi_peak_hours)
    cols = min(6, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 5, rows * 5))
    axes = np.array(axes).reshape(-1)  # 展平以支援單行也能 index
    for i in range(n, len(axes)):
        axes[i].axis("off")

    if len(multi_peak_hours) == 1:
        axes = [axes]

    for i, hour in enumerate(multi_peak_hours):
        ax = axes[i]

        real_data = df_real[df_real["hour"] == hour]["TravelSpeed"]
        sim_data = df_sim[df_sim["hour"] == hour]["預測車速"]

        sns.histplot(real_data, kde=False, bins=30, stat="count",
                     ax=ax, color=color, edgecolor='black', alpha=0.7, label="真實")

        mu = sim_data.mean()
        sigma = sigma_by_time.get(f"{hour:02d}:00", 3)
        lower, upper = (mu - 4 * sigma - mu) / sigma, (mu + 4 * sigma - mu) / sigma

        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
        x = np.clip(x, 0, 120)
        trunc_pdf = truncnorm.pdf(x, lower, upper, loc=mu, scale=sigma)
        bin_width = (real_data.max() - real_data.min()) / 30
        scaled_pdf = trunc_pdf * len(real_data) * bin_width

        ax.plot(x, scaled_pdf, color='darkgreen', linewidth=1.5, label="模擬截尾分布")
        ax.set_xlim(0, 120)
        ax.set_title(f"{hour:02d}:00 - {hour+1:02d}:00\n⚠️ 多峰", fontsize=12)
        ax.set_xlabel("車速 (km/h)")
        ax.set_ylabel("車輛數")
        ax.legend()

        ax.text(0.02, 0.95, f"μ={mu:.1f}, σ={sigma:.1f}", transform=ax.transAxes, fontsize=9)

    fig.suptitle(f"⚠️ {direction}方向 多峰車速分布時段（真實直方圖 + 模擬截尾 PDF）", fontsize=18)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f"multi_peak_hours_{direction}.png", dpi=300)
    plt.show()
print("畫第二種圖...")
#plot_multi_peak_hours(df_all, df_sim, simulator.sigma_by_time, direction="南下")
#plot_multi_peak_hours(df_all, df_sim, simulator.sigma_by_time, direction="北上")

# 6️⃣ 匯出結果
print("輸出結果中...")
#simulator.export_vehicle_time_info("vehicle_passage_time.csv")
