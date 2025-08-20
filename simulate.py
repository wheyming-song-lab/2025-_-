import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import timedelta
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# === 基本參數 ===
RID = "0024"
STRATEGY = "model_based_hybrid"

# —— 控制強度（更保守的放行） ——
CYCLE_SEC     = 90.0   # 由 60 -> 90，5 分鐘只剩 3 個 cycle（放行次數↓）
SAT_FLOW_VPS  = 0.40   # 由 0.50 -> 0.40，飽和流率下降（輛/秒）
RAMP_LANES    = 1

# —— 匝道實體容量（限制隊列、超量即觸發分流/改道） ——
RAMP_STORAGE_LEN_M = 500      # 匝道可排隊長度(公尺)；若未知可先用 500m
STOPPED_HEADWAY_M  = 7.5      # 壅塞下靜止車距(含車長)
RAMP_STORAGE_MAX   = int(np.floor(RAMP_LANES * RAMP_STORAGE_LEN_M / STOPPED_HEADWAY_M))  # ≈ 66 輛/單車道500m

# —— 速度模型（避免過度樂觀） ——
ALPHA_SPEED   = 0.03
V_FREE        = 100.0
K_TARGET_PCTL = 0.60
DENSITY_MARGIN= 1.20
KJAM_FACTOR   = 4.0
BETA_RECOVER  = 0.15

# —— 低速止血機制 ——
LOW_SPEED_THRESH        = 50.0
LOW_SPEED_WINDOW        = 2
LOW_SPEED_RELEASE_RATIO = 0.30
LOW_SPEED_G_MIN         = 8.0

# —— 依壅塞機率設定「當期最大放行比例」的保險絲 ——
ENABLE_RATIO_CAP = True
RATIO_CAPS = {
    0.80: 0.40,
    0.60: 0.60,
    0.40: 0.75,
    0.00: 0.90,
}

# —— 主線容量提醒門檻（以密度為準） ——  # <<< 新增
MAIN_TARGET_UTIL = 1.00  # k_next / K_TARGET 達到此比例→提醒
MAIN_JAM_UTIL    = 0.55  # k_next / K_JAM 達到此比例（已超容量頂點0.5附近）→提醒
MAIN_SPEED_LOW   = 50.0  # 一般提醒速度門檻
MAIN_SPEED_CRIT  = 40.0  # 強烈提醒速度門檻

PLOT = True

# === 檔案路徑 ===
VEHICLE_CSV = r"C:\Users\User\Desktop\交通部\交通部競賽\new_train\xgboost\v5模擬\sim_0024_outputs\figs\vehicles_0024_24h_profile.csv"
MODEL_DIR   = r"C:\Users\User\Desktop\交通部\交通部競賽\new_train\models\models_ensemble"
OUT_DIR     = r"C:\Users\User\Desktop\交通部\交通部競賽\new_train\xgboost\v5模擬\sim_0024_outputs"

# === 特徵欄位 ===
FEATURE_COLS = [
    "traffic_density", "density_diff", "drop_speed", "speed_ma5", "speed_ratio",
    "flow_var", "nonlinear_inter", "flow_ma5", "flow_slope", "flow_drop", "speed_diff"
]

# === 建立輸出資料夾 ===
def ensure_outdir(p): Path(p).mkdir(parents=True, exist_ok=True); return Path(p)

# === 載入模型 ===
def load_models(model_dir, rid):
    paths = {
        "lstm_A": Path(model_dir) / f"lstm_A_{rid}.keras",
        "lstm_B": Path(model_dir) / f"lstm_B_{rid}.keras",
        "xgb_A":  Path(model_dir) / f"xgb_A_{rid}.pkl",
        "xgb_B":  Path(model_dir) / f"xgb_B_{rid}.pkl",
        "scaler_lstm":     Path(model_dir) / f"scaler_lstm_{rid}.pkl",
        "scaler_feat_A":   Path(model_dir) / f"scaler_feat_A_{rid}.pkl",
        "scaler_feat_B":   Path(model_dir) / f"scaler_feat_B_{rid}.pkl",
    }
    return (
        load_model(paths["lstm_A"]),
        load_model(paths["lstm_B"]),
        joblib.load(paths["xgb_A"]),
        joblib.load(paths["xgb_B"]),
        joblib.load(paths["scaler_lstm"]),
        joblib.load(paths["scaler_feat_A"]),
        joblib.load(paths["scaler_feat_B"]),
    )

# === 讀取車單並聚合成 5 分鐘時序 ===
def build_5min_timeseries_from_vehicle(csv_path):
    df = pd.read_csv(csv_path)
    df["t2_start"]  = pd.to_datetime(df["進入時間"], errors="coerce")
    df["v2_expect"] = pd.to_numeric(df["期望速度(km/h)"], errors="coerce")
    df["src"]       = df.get("來源", "主線").astype(str)
    df = df.dropna(subset=["t2_start","v2_expect"])
    df["time_bin"] = df["t2_start"].dt.floor("5min")
    is_ramp = df["src"].str.contains("匝|ramp|Ramp", case=False, na=False)
    flow_total = df.groupby("time_bin")["車輛ID"].count().rename("flow_total").astype(int)
    flow_ramp  = df[is_ramp].groupby("time_bin")["車輛ID"].count().rename("flow_ramp").astype(int)
    flow_main  = df[~is_ramp].groupby("time_bin")["車輛ID"].count().rename("flow_main").astype(int)
    speed_base = df.groupby("time_bin")["v2_expect"].mean().rename("speed_base")
    ts = pd.concat([flow_total, flow_main, flow_ramp, speed_base], axis=1).sort_index().fillna(0)

    # 尖峰時段提前 15 分鐘：06:45–09:00、16:45–19:00
    min_of_day = ts.index.hour * 60 + ts.index.minute
    am_start, am_end = 7*60 - 15, 9*60
    pm_start, pm_end = 17*60 - 15, 19*60
    ts["is_peak"] = (
        ((min_of_day >= am_start) & (min_of_day < am_end)) |
        ((min_of_day >= pm_start) & (min_of_day < pm_end))
    ).astype(int)

    ts["hour"] = ts.index.hour
    return ts

# === 特徵工程 ===
def build_features(ts):
    df = ts.copy()
    df["TravelSpeed"] = df["speed_base"]
    df["總車輛數"] = df["flow_total"]
    df["flow_ma5"]   = df["flow_total"].rolling(5, min_periods=1).mean()
    df["flow_slope"] = df["flow_ma5"].diff().fillna(0)
    df["flow_var"]   = df["flow_total"].rolling(5, min_periods=1).std().fillna(0)
    df["flow_drop"]  = df["flow_ma5"] - df["flow_total"]
    df["speed_ma5"]  = df["TravelSpeed"].rolling(5, min_periods=1).mean()
    df["speed_diff"] = df["TravelSpeed"].diff().fillna(0)
    df["drop_speed"] = df["TravelSpeed"].diff().fillna(0)
    df["traffic_density"] = df["flow_total"] / (df["TravelSpeed"] + 1e-5)
    df["density_diff"]    = df["traffic_density"].diff().fillna(0)
    df["speed_ratio"]     = df["speed_ma5"] / (df["TravelSpeed"].rolling(15, min_periods=1).mean() + 1e-5)
    df["nonlinear_inter"] = df["drop_speed"] * df["density_diff"]
    df["lstm_pred_speed"] = np.nan
    return df

# === 建立 LSTM 輸入序列 ===
def create_lstm_sequences(df_for_lstm, feature_cols, window):
    X = [df_for_lstm[feature_cols].iloc[i:i+window].values for i in range(len(df_for_lstm)-window)]
    X = np.array(X)
    base_speed_norm = df_for_lstm["TravelSpeed"].values[window-1:]
    start_index = window
    return X, base_speed_norm, start_index

# === 混合判斷控制策略（含容量限制 / 超量分流 / 改道提醒） ===
def signal_model_based_hybrid(df):
    green_sec = []
    last_g = None
    high_flow_thr = 100
    low_flow_thr  = 70
    strict_green  = 10.0
    relax_green   = 50.0
    hyst_on_count = 0
    hyst_off_count = 0
    active_strict = False

    queue = 0.0
    queue_list, ramp_after, total_after, speed_after, cut_ratio = [], [], [], [], []
    diverted_list = []           # 超容量而分流的車數
    alert_flag, alert_speed, alert_msg = [], [], []  # 改道提醒
    k_list, util_t_list, util_j_list = [], [], []    # <<< 新增：主線密度與利用率追蹤

    cycles = int(np.floor(300.0 / CYCLE_SEC))
    cap_per_sec = SAT_FLOW_VPS * RAMP_LANES

    # 動態估計可接受密度上限 K_TARGET
    dens_all = (df["flow_main"] / (df["speed_base"] + 1e-5)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(dens_all) == 0:
        K_TARGET = 1e6
    else:
        K_TARGET = DENSITY_MARGIN * np.percentile(dens_all, int(K_TARGET_PCTL*100))
    K_JAM = max(K_TARGET * KJAM_FACTOR, K_TARGET + 1e-5)
    low_speed_count = 0

    for idx in range(len(df)):
        p = df["p_cong"].iloc[idx]
        f_main = df["flow_main"].iloc[idx]
        f_ramp = df["flow_ramp"].iloc[idx]
        speed_base = df["speed_base"].iloc[idx]

        # 提前 15 分鐘尖峰：若 p < 0.4 也提升至 0.4 啟動控制
        if df["is_peak"].iloc[idx] == 1 and p < 0.4:
            p = 0.4

        # 基礎綠秒
        if p >= 0.8:   g = 10.0
        elif p >= 0.6: g = 20.0
        elif p >= 0.4: g = 30.0
        elif p >= 0.2: g = 40.0
        else:          g = 50.0

        # 高流量滯後嚴控
        if p >= 0.4 and f_main >= high_flow_thr:
            hyst_on_count += 1; hyst_off_count = 0
        else:
            hyst_off_count += 1; hyst_on_count = 0
        if not active_strict and hyst_on_count >= 2:
            active_strict = True
        elif active_strict and hyst_off_count >= 2:
            active_strict = False
        if active_strict:
            g = min(g, strict_green)

        # 低流量放寬
        if p < 0.2 and f_main <= low_flow_thr:
            g = relax_green

        # 匝道排隊保護（注意：容量 66，這條 80 的門檻在此案例不會觸發）
        if queue > 80:
            g = max(g, 30.0)

        # 綠秒平滑
        if last_g is not None:
            if g > last_g + 5: g = last_g + 5
            elif g < last_g - 5: g = last_g - 5
        last_g = g
        green_sec.append(g)

        # 物理放行上限（週期 * 飽和流率 * 綠秒）
        release_cap = cycles * cap_per_sec * g
        release_now = min(release_cap, f_ramp + queue)

        # (1) 控密度：強制 k_next 不超過 K_TARGET
        max_release_by_density = max(0.0, K_TARGET * max(speed_base, 1e-5) - f_main)
        release_now = min(release_now, max_release_by_density)

        # (2) 比例保險絲
        if ENABLE_RATIO_CAP:
            for thr, ratio in sorted(RATIO_CAPS.items(), reverse=True):
                if p >= thr:
                    release_now = min(release_now, (f_ramp + queue) * ratio)
                    break

        # (3) 低速止血
        if speed_base < LOW_SPEED_THRESH:
            low_speed_count += 1
        else:
            low_speed_count = 0
        if low_speed_count >= LOW_SPEED_WINDOW:
            release_now = min(release_now, LOW_SPEED_RELEASE_RATIO * (f_ramp + queue))
            g_tmp_for_release = min(g, LOW_SPEED_G_MIN)
            release_cap_tmp = cycles * cap_per_sec * g_tmp_for_release
            release_now = min(release_now, release_cap_tmp)

        # —— 更新排隊（容量限制 + 溢出分流） ——
        release_now = max(0.0, release_now)
        queue_next_raw = queue + f_ramp - release_now

        overflow_now = max(0.0, queue_next_raw - RAMP_STORAGE_MAX)  # 超出容量
        diverted_now = overflow_now
        queue = min(RAMP_STORAGE_MAX, max(0.0, queue_next_raw))

        # 切車量（當期未放行）
        cut_cars = max(0.0, f_ramp - release_now)

        # 放行後總流量
        flow_tot_after = f_main + release_now

        # 速度更新：基本圖（Greenshields）+ 復原 + 切車效應
        k_next = (f_main + release_now) / max(speed_base, 1e-5)
        v_gre = V_FREE * max(0.0, 1.0 - k_next / max(K_JAM, 1e-5))
        v_after = 0.5 * speed_base + 0.5 * (speed_base + BETA_RECOVER * (v_gre - speed_base))
        v_after += ALPHA_SPEED * cut_cars
        v_after = float(np.clip(v_after, 0.0, V_FREE))

        # === 主線容量版改道提醒（以密度判定） ===  # <<< 修改
        k_list.append(k_next)
        util_t = k_next / max(K_TARGET, 1e-5)   # 相對目標密度
        util_j = k_next / max(K_JAM, 1e-5)      # 相對壅塞密度
        util_t_list.append(util_t)
        util_j_list.append(util_j)

        severe = (
            (util_j >= MAIN_JAM_UTIL and v_after < MAIN_SPEED_LOW) or   # 超容量門檻且速度偏低
            (util_t >= MAIN_TARGET_UTIL and p >= 0.6) or                 # 超過目標密度且壅塞風險高
            (v_after < MAIN_SPEED_CRIT)                                  # 速度極低
        )
        if severe:
            alert_flag.append(1)
            alert_speed.append(v_after)
            if v_after < MAIN_SPEED_CRIT:
                msg = "主線極度壅塞（<40 km/h），請改道。"
            elif util_j >= MAIN_JAM_UTIL and v_after < MAIN_SPEED_LOW:
                msg = f"主線密度已接近/超過容量門檻，且速度偏低，建議改道。"
            else:
                msg = "主線密度高於目標且壅塞風險高，建議改道。"
            alert_msg.append(msg)
        else:
            alert_flag.append(0)
            alert_speed.append(np.nan)
            alert_msg.append("")

        # —— 記錄 ——
        ramp_after.append(int(round(release_now)))
        total_after.append(int(round(flow_tot_after)))
        speed_after.append(v_after)
        cut_ratio.append(0.0 if f_ramp <= 1e-9 else float(np.clip(1.0 - (release_now / f_ramp), 0.0, 1.0)))
        queue_list.append(queue)
        diverted_list.append(int(round(diverted_now)))

    df["signal_green_sec"] = green_sec
    df["signal_queue"] = queue_list
    df["flow_ramp_after"] = ramp_after
    df["flow_total_after"] = total_after
    df["speed_after"] = speed_after
    df["ramp_cut_ratio"] = cut_ratio
    df["ramp_diverted"] = diverted_list

    # 改道提醒輸出欄位（主線容量版）  # <<< 新增
    df["reroute_alert"]   = alert_flag
    df["reroute_speed"]   = alert_speed
    df["reroute_message"] = alert_msg
    df["k_main"]       = k_list
    df["util_target"]  = util_t_list
    df["util_jam"]     = util_j_list
    return df

# === 固定 24 小時圖軸 ===
def setup_24h_axis(ax, dt_index):
    if len(dt_index) == 0:
        return
    day_start = dt_index.min().normalize()
    day_end = day_start + timedelta(days=1)
    ax.set_xlim(day_start, day_end)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True)

# === 主程式 ===
def run_simulation():
    out_dir = ensure_outdir(OUT_DIR)
    lstm_A, lstm_B, xgb_A, xgb_B, scaler_lstm, scaler_feat_A, scaler_feat_B = load_models(MODEL_DIR, RID)

    ts = build_5min_timeseries_from_vehicle(VEHICLE_CSV)
    df = build_features(ts)

    # LSTM 預測
    df_lstmA = df.copy()
    df_lstmA["TravelSpeed"] = scaler_lstm.transform(df_lstmA[["TravelSpeed"]])
    feat_A_all = scaler_feat_A.transform(df_lstmA[FEATURE_COLS])
    X_lstm, base_speed_norm, start_idx = create_lstm_sequences(df_lstmA, FEATURE_COLS, 24)
    dv_pred = lstm_A.predict(X_lstm, verbose=0).flatten()
    v_pred_norm = dv_pred + base_speed_norm[:len(dv_pred)]
    v_pred_real = scaler_lstm.inverse_transform(v_pred_norm.reshape(-1,1)).flatten()
    df.iloc[start_idx:start_idx+len(v_pred_real), df.columns.get_loc("lstm_pred_speed")] = v_pred_real

    # XGB 預測壅塞機率（尖峰使用 B 模型，且 is_peak 已提前 15 分鐘）
    feat_B_all = scaler_feat_B.transform(df[FEATURE_COLS])
    valid_mask = ~df["lstm_pred_speed"].isna()
    dfv = df.loc[valid_mask].copy()
    pos = np.where(valid_mask.values)[0]
    def build_xgb_X(X_feat, dfx):
        other = dfx[["lstm_pred_speed","總車輛數","is_peak"]].values
        return np.hstack([X_feat, other])
    X_A = build_xgb_X(feat_A_all[pos,:], dfv)
    X_B = build_xgb_X(feat_B_all[pos,:], dfv)
    proba_A = xgb_A.predict_proba(X_A)[:,1]
    proba_B = xgb_B.predict_proba(X_B)[:,1]
    is_peak_v = dfv["is_peak"].astype(bool).values
    p_cong = np.where(is_peak_v, proba_B, proba_A)
    df.loc[valid_mask,"p_cong"] = p_cong
    df["p_cong"] = df["p_cong"].fillna(0.0)

    # 策略
    df = signal_model_based_hybrid(df)

    # 指標
    df["congested_base"]  = (df["speed_base"]  < 60).astype(int)
    df["congested_after"] = (df["speed_after"] < 60).astype(int)
    def pct(x): return 100.0 * float(x)
    metrics = {
        "avg_speed_before": float(df["speed_base"].mean()),
        "avg_speed_after": float(df["speed_after"].mean()),
        "cong_rate_before_pct": float(pct(df["congested_base"].mean())),
        "cong_rate_after_pct": float(pct(df["congested_after"].mean())),
        "avg_ramp_cut_pct": float(pct(df["ramp_cut_ratio"].fillna(0).mean())),
        "total_flow_before": int(df["flow_total"].sum()),
        "total_flow_after": int(df["flow_total_after"].sum()),
        "avg_green_sec": float(df["signal_green_sec"].mean()),
        "max_queue": int(df["signal_queue"].max()),
    }

    out_csv = Path(out_dir) / f"sim_{RID}_{STRATEGY}.csv"
    df.to_csv(out_csv, encoding="utf-8-sig")

    # 另存「改道提醒」清單（中文欄名）
    alerts = df[df["reroute_alert"] == 1].copy()
    if not alerts.empty:
        alerts_out = alerts.assign(time_bin=alerts.index)[
            ["time_bin", "k_main", "util_target", "util_jam", "p_cong", "speed_after", "reroute_message"]
        ].rename(columns={
            "time_bin": "時間",
            "k_main": "主線密度",
            "util_target": "相對目標密度",
            "util_jam": "相對壅塞密度",
            "p_cong": "壅塞機率",
            "speed_after": "施策後速度(km/h)",
            "reroute_message": "建議"
        })
        alerts_csv = Path(out_dir) / f"alerts_{RID}_{STRATEGY}.csv"
        alerts_out.to_csv(alerts_csv, index=False, encoding="utf-8-sig")
        print(f"\n提醒清單已輸出：{alerts_csv}（共 {alerts_out.shape[0]} 筆）")
    else:
        print("\n本日未觸發改道提醒。")

    print("\n=== 模擬完成（控密度 + 基本圖回復 + 止血 + 保險絲 + 匝道保護 + 尖峰提前15分 + 容量限制 + 改道提醒-主線容量） ===")
    for k, v in metrics.items():
        if "pct" in k: print(f"{k}: {v:.2f}%")
        elif "speed" in k: print(f"{k}: {v:.2f} km/h")
        elif "green" in k: print(f"{k}: {v:.2f} s")
        else: print(f"{k}: {v}")
    print(f"\nCSV 已輸出：{out_csv}")

    if PLOT:
        fig_dir = Path(out_dir) / "figs"; fig_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df.index, df["speed_base"], label="速度(前)")
        ax.plot(df.index, df["speed_after"], label="速度(後)")
        ax.axhline(60, color="red", ls="--", alpha=0.7)
        ax.set_title(f"{RID} 速度：施策前 vs 後（{STRATEGY}）")
        ax.set_xlabel("時間"); ax.set_ylabel("km/h"); ax.legend()

        # ★ y 軸 0~100，每 10 一格
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 10))

        setup_24h_axis(ax, df.index); plt.tight_layout()
        plt.savefig(fig_dir / f"{RID}_speed_{STRATEGY}.png", dpi=200); plt.close(fig)

        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df.index, df["flow_total"], label="總流量(前)")
        ax.plot(df.index, df["flow_total_after"], label="總流量(後)")
        ax.set_title(f"{RID} 流量：施策前 vs 後（{STRATEGY}）")
        ax.set_xlabel("時間"); ax.set_ylabel("輛/5分鐘"); ax.legend()
        setup_24h_axis(ax, df.index); plt.tight_layout()
        plt.savefig(fig_dir / f"{RID}_flow_{STRATEGY}.png", dpi=200); plt.close(fig)

        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df.index, df["signal_green_sec"], label="綠秒/箱")
        ax.set_title(f"{RID} 綠燈秒數（{STRATEGY}）")
        ax.set_xlabel("時間"); ax.set_ylabel("秒"); ax.legend()
        setup_24h_axis(ax, df.index); plt.tight_layout()
        plt.savefig(fig_dir / f"{RID}_green_{STRATEGY}.png", dpi=200); plt.close(fig)

        print(f"圖片已輸出：{fig_dir}")

    return metrics, out_csv

if __name__ == "__main__":
    run_simulation()
