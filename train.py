# === train_ensemble_improved_v2_SAVE_LSTM_BLOCKS_FINALTEST2025.py ===
# 需求對齊：
# - 以「天」為單位做 Block：每路段 5 個 blocks
# - 每個 block 內：2024 資料 70% 訓練、10% 驗證、20% 測試（僅在 2024 範圍內切）
# - 模型訓練只用 2024 的「訓練集」；縮放器亦只在訓練集上 fit
# - 2025 資料作為「最終測試」（final test）；每路段彙整 5 個 block 的成績 → 平均 ± 標準誤（小數點後 1 位）
# - 壅塞規則：v<60；或 v<70 且 v/c≥0.9（等價 q≥0.9C），再加上持續性（predict_shift=3 中至少 2/3 成立）
# - 車道數固定為 4

import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from xgboost import XGBClassifier
from holidays import holiday_dates
from tqdm import trange

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== 參數 =====
PREDICT_SHIFT = 3
WINDOW = 24
N_BLOCKS = 5
SPLIT_RATIOS = (0.7, 0.1, 0.2)   # train / val / test
SEED_BASE = 20240816

# === 容量假設 ===
N_LANES = 4
CAP_VPH_PER_LANE = 2000
C_VPH = N_LANES * CAP_VPH_PER_LANE
QPF_TO_VPH = 12.0  # 5分鐘轉時流量

# ===== I/O =====
model_dir  = r"C:\Users\User\Desktop\交通部\交通部競賽\new_train\models\models_ensemble"
output_dir = r"C:\Users\User\Desktop\交通部\交通部競賽\output_ensemble"
blocks_dir = os.path.join(output_dir, "blocks")
data_2024  = r"C:\Users\User\Desktop\交通部\交通部競賽\data_2024"
data_2025  = r"C:\Users\User\Desktop\交通部\交通部競賽\data_2025"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(blocks_dir, exist_ok=True)

def save_path(name):
    return os.path.join(model_dir, name)

roads = {
    "0021": "0021_圓山交流道_到_台北交流道_final.xlsx",
    "0022": "0022_台北交流道_到_圓山交流道_final.xlsx",
    "0023": "0023_台北交流道_到_三重交流道_final.xlsx",
    "0024": "0024_三重交流道_到_台北交流道_final.xlsx",
}

holiday_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in holiday_dates]

# ===== 前處理 =====
def filter_data(df, remove_peak):
    df = df.copy()
    df["date"] = df["時間點"].dt.date
    df["weekday"] = df["時間點"].dt.weekday
    df["hour"] = df["時間點"].dt.hour
    # 平日、去假日
    df = df[~df["date"].isin(holiday_dates)]
    df = df[df["weekday"] < 5]
    if remove_peak:
        df = df[~((df["hour"].between(7, 9)) | (df["hour"].between(17, 19)))]
    return df

# ===== 壅塞規則（單點） =====
def is_congested_point(v, q_5min):
    if v < 60:
        return True
    q_vph = (q_5min or 0.0) * QPF_TO_VPH
    if (v < 70) and (q_vph >= 0.9 * C_VPH):
        return True
    return False

# ===== 持續性標籤（未來 PREDICT_SHIFT 視窗至少 2/3 成立） =====
def build_targets_with_persistence(df, predict_shift=PREDICT_SHIFT):
    df = df.copy()
    cond = df[["TravelSpeed", "總車輛數"]].apply(
        lambda s: is_congested_point(float(s["TravelSpeed"]), float(s["總車輛數"])),
        axis=1
    ).astype(int).values

    n = len(cond)
    future_hits = np.zeros(n, dtype=int)
    k_need = int(np.ceil((2/3) * predict_shift))

    for i in range(n):
        j_start = i + 1
        j_end = min(i + predict_shift, n - 1)
        if j_start <= j_end:
            future_hits[i] = cond[j_start:j_end+1].sum()
        else:
            future_hits[i] = 0

    target_warning = (future_hits >= k_need).astype(int)
    return target_warning

# ===== 特徵工程 =====
def build_features(df, predict_shift=PREDICT_SHIFT, lstm_pred=None):
    df = df.copy()
    df["flow_ma5"]  = df["總車輛數"].rolling(5, min_periods=1).mean()
    df["flow_slope"]= df["flow_ma5"].diff().fillna(0)
    df["flow_var"]  = df["總車輛數"].rolling(5, min_periods=1).std().fillna(0)
    df["flow_drop"] = df["flow_ma5"] - df["總車輛數"]

    df["speed_ma5"] = df["TravelSpeed"].rolling(5, min_periods=1).mean()
    df["speed_diff"]= df["TravelSpeed"].diff().fillna(0)
    df["drop_speed"]= df["TravelSpeed"].diff().fillna(0)

    df["traffic_density"] = df["總車輛數"] / (df["TravelSpeed"] + 1e-5)
    df["density_diff"]    = df["traffic_density"].diff().fillna(0)
    df["speed_ratio"]     = df["speed_ma5"] / (df["TravelSpeed"].rolling(15, min_periods=1).mean() + 1e-5)

    df["nonlinear_inter"] = df["drop_speed"] * df["density_diff"]
    df["is_peak"] = ((df["hour"].between(7, 9)) | (df["hour"].between(17, 19))).astype(int)

    if lstm_pred is None:
        df["lstm_pred_speed"] = np.nan
    else:
        fill_len = min(len(lstm_pred), len(df))
        df["lstm_pred_speed"] = np.nan
        df.iloc[:fill_len, df.columns.get_loc("lstm_pred_speed")] = lstm_pred[:fill_len]

    df["target_warning"] = build_targets_with_persistence(df, predict_shift=predict_shift)
    return df

# ===== LSTM 資料組裝 =====
def create_lstm_data(df, features, window=WINDOW, shift=PREDICT_SHIFT):
    X, y, v_now = [], [], df["TravelSpeed"].values
    for i in range(len(df) - window - shift):
        X.append(df[features].iloc[i:i+window].values)
        v1 = v_now[i+window-1]
        v2 = v_now[i+window+shift-1]
        y.append(v2 - v1)
    return np.array(X), np.array(y), v_now[window-1: -shift]

def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        LSTM(64, kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss=Huber())
    return model

# ===== XGB =====
def train_xgb(df, features, weighted):
    df_ = df.dropna(subset=features + ["target_warning"]).copy()
    X = df_[features].values
    y = df_["target_warning"].values
    weights = np.ones(len(df_))
    if weighted:
        weights[(df_["is_peak"] == 0) & (df_["target_warning"] == 1)] = 3.0
    model = XGBClassifier(
        n_estimators=50, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=2.0, eval_metric="logloss", random_state=42
    )
    model.fit(X, y, sample_weight=weights)
    return model

def plot_confusion(y_true, y_pred, out, rid, block_id):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["無壅塞", "壅塞前兆"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"模型集成 - 路段 {rid} - Block {block_id} (Final 2025)")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

# ===== Block 產生/讀取（以 2024 的「天」做切分） =====
def get_block_file(rid):
    return os.path.join(blocks_dir, f"blocks_{rid}.json")

def get_or_create_blocks(rid, dates_2024):
    f = get_block_file(rid)
    if os.path.exists(f):
        with open(f, "r", encoding="utf-8") as fh:
            return json.load(f)

    dates = sorted(list(set(dates_2024)))
    n = len(dates)
    blocks = []
    for b in range(N_BLOCKS):
        rng = np.random.default_rng(SEED_BASE + b)
        idx = np.arange(n)
        rng.shuffle(idx)
        d_shuf = [dates[i] for i in idx]

        n_tr = int(SPLIT_RATIOS[0] * n)
        n_va = int(SPLIT_RATIOS[1] * n)

        train_dates = d_shuf[:n_tr]
        val_dates   = d_shuf[n_tr:n_tr+n_va]
        test_dates  = d_shuf[n_tr+n_va:]

        blocks.append({
            "train": [str(d) for d in train_dates],
            "val":   [str(d) for d in val_dates],
            "test":  [str(d) for d in test_dates],
        })

    with open(f, "w", encoding="utf-8") as fh:
        json.dump(blocks, fh, ensure_ascii=False, indent=2)
    return blocks

# ===== 特徵欄位 =====
feature_cols = [
    "traffic_density", "density_diff", "drop_speed", "speed_ma5", "speed_ratio",
    "flow_var", "nonlinear_inter", "flow_ma5", "flow_slope", "flow_drop", "speed_diff"
]
xgb_cols = feature_cols + ["lstm_pred_speed", "總車輛數", "is_peak"]

# ===== 主流程：每路段 x 每 block，訓練於 2024(train)，最終測試於 2025 =====
summary_rows = []
agg_rows = []

for rid, fname in roads.items():
    print(f"\n🚦 路段 {rid} | 取得 2024 資料與建立 blocks ...")
    df24 = pd.read_excel(os.path.join(data_2024, fname))
    # 保留兩個版本（非尖峰/全時段）供 A/B 模型各自取「訓練集」
    df24_A = filter_data(df24, remove_peak=True)
    df24_B = filter_data(df24, remove_peak=False)

    # 用 2024 的「天」建立/讀取 blocks（以 union 日期切，確保 A/B 可對齊日期集合）
    all_dates_2024 = set(df24_A["date"]).union(set(df24_B["date"]))
    blocks = get_or_create_blocks(rid, all_dates_2024)
    print(f"   -> blocks 數量：{len(blocks)}（每個 block 70/10/20）")

    # 讀 2025 做最終測試（固定）
    df25_raw = pd.read_excel(os.path.join(data_2025, fname))
    df25 = filter_data(df25_raw, remove_peak=False)
    df25["TravelSpeed_real"] = df25["TravelSpeed"].copy()

    block_scores = []
    for b_id, b in enumerate(blocks, start=1):
        print(f"\n🧱 {rid} - Block {b_id}：訓練(2024-train) → 最終測試(2025)")

        tr_dates = set(pd.to_datetime(b["train"]).date)
        va_dates = set(pd.to_datetime(b["val"]).date)
        te_dates = set(pd.to_datetime(b["test"]).date)

        # 2024 訓練集（只用 train，依需求）
        df24_A_tr = df24_A[df24_A["date"].isin(tr_dates)].copy()
        df24_B_tr = df24_B[df24_B["date"].isin(tr_dates)].copy()

        # （保留不使用但產出以便你後續調參）
        df24_A_va = df24_A[df24_A["date"].isin(va_dates)].copy()
        df24_B_va = df24_B[df24_B["date"].isin(va_dates)].copy()
        df24_A_te = df24_A[df24_A["date"].isin(te_dates)].copy()
        df24_B_te = df24_B[df24_B["date"].isin(te_dates)].copy()

        # 特徵與標籤（持續性）
        df24_A_tr = build_features(df24_A_tr, predict_shift=PREDICT_SHIFT)
        df24_B_tr = build_features(df24_B_tr, predict_shift=PREDICT_SHIFT)
        # 可視需要也對 val/test 做 build_features（這裡先保留供你之後驗證用）
        _ = build_features(df24_A_va, predict_shift=PREDICT_SHIFT)
        _ = build_features(df24_B_va, predict_shift=PREDICT_SHIFT)
        _ = build_features(df24_A_te, predict_shift=PREDICT_SHIFT)
        _ = build_features(df24_B_te, predict_shift=PREDICT_SHIFT)

        # 縮放器（速度）：只在 A 的訓練集上 fit（沿用你的設計）
        scaler_lstm = MinMaxScaler()
        df24_A_tr["TravelSpeed"] = scaler_lstm.fit_transform(df24_A_tr[["TravelSpeed"]])
        df24_B_tr["TravelSpeed"] = scaler_lstm.transform(df24_B_tr[["TravelSpeed"]])

        # 特徵縮放器：A/B 各自用訓練集 fit
        feature_scaler_A = MinMaxScaler().fit(df24_A_tr[feature_cols])
        feature_scaler_B = MinMaxScaler().fit(df24_B_tr[feature_cols])
        df24_A_tr[feature_cols] = feature_scaler_A.transform(df24_A_tr[feature_cols])
        df24_B_tr[feature_cols] = feature_scaler_B.transform(df24_B_tr[feature_cols])

        # 落地縮放器（含 block id）
        joblib.dump(scaler_lstm,      save_path(f"scaler_lstm_{rid}_B{b_id}.pkl"))
        joblib.dump(feature_scaler_A, save_path(f"scaler_feat_A_{rid}_B{b_id}.pkl"))
        joblib.dump(feature_scaler_B, save_path(f"scaler_feat_B_{rid}_B{b_id}.pkl"))

        # LSTM 訓練資料（只用 2024-train）
        X_A, y_A, v_A = create_lstm_data(df24_A_tr, feature_cols, window=WINDOW, shift=PREDICT_SHIFT)
        X_B, y_B, v_B = create_lstm_data(df24_B_tr, feature_cols, window=WINDOW, shift=PREDICT_SHIFT)

        # 訓練 LSTM A/B
        lstm_A = build_lstm((X_A.shape[1], X_A.shape[2]))
        print(f"   -> 訓練 LSTM A（{rid}, B{b_id}）")
        for _ in trange(30, desc=f"LSTM A {rid}-B{b_id}"):
            lstm_A.fit(X_A, y_A, epochs=1, batch_size=64, verbose=0)

        lstm_B = build_lstm((X_B.shape[1], X_B.shape[2]))
        print(f"   -> 訓練 LSTM B（{rid}, B{b_id}）")
        for _ in trange(30, desc=f"LSTM B {rid}-B{b_id}"):
            lstm_B.fit(X_B, y_B, epochs=1, batch_size=64, verbose=0)

        # 儲存 LSTM
        lstm_A_path = save_path(f"lstm_A_{rid}_B{b_id}.keras")
        lstm_B_path = save_path(f"lstm_B_{rid}_B{b_id}.keras")
        lstm_A.save(lstm_A_path)
        lstm_B.save(lstm_B_path)

        # 用訓練集做內推，產生 lstm_pred_speed（只給 XGB 訓練用）
        y_pred_A = lstm_A.predict(X_A, verbose=0).flatten() + v_A[:len(y_A)]
        y_pred_B = lstm_B.predict(X_B, verbose=0).flatten() + v_B[:len(y_B)]
        y_pred_A_real = scaler_lstm.inverse_transform(y_pred_A.reshape(-1, 1)).flatten()
        y_pred_B_real = scaler_lstm.inverse_transform(y_pred_B.reshape(-1, 1)).flatten()

        df24_A_tr_lstm = df24_A_tr.copy()
        df24_B_tr_lstm = df24_B_tr.copy()
        df24_A_tr_lstm["lstm_pred_speed"] = np.append(y_pred_A_real, [np.nan] * (len(df24_A_tr_lstm) - len(y_pred_A_real)))
        df24_B_tr_lstm["lstm_pred_speed"] = np.append(y_pred_B_real, [np.nan] * (len(df24_B_tr_lstm) - len(y_pred_B_real)))

        # 訓練 XGB（訓練集）
        print(f"   -> 訓練 XGB A（{rid}, B{b_id}）")
        xgb_A = train_xgb(df24_A_tr_lstm, xgb_cols, weighted=True)
        print(f"   -> 訓練 XGB B（{rid}, B{b_id}）")
        xgb_B = train_xgb(df24_B_tr_lstm, xgb_cols, weighted=False)

        # 儲存 XGB
        joblib.dump(xgb_A, save_path(f"xgb_A_{rid}_B{b_id}.pkl"))
        joblib.dump(xgb_B, save_path(f"xgb_B_{rid}_B{b_id}.pkl"))

        # ===== 最終測試（2025） =====
        print(f"   -> 最終測試 2025（{rid}, B{b_id}）")
        df_t = df25.copy()
        df_t = build_features(df_t, predict_shift=PREDICT_SHIFT)

        # 速度縮放（以 2024-train 的 A-scaler）
        df_t["TravelSpeed"] = scaler_lstm.transform(df_t[["TravelSpeed"]])

        # 特徵縮放（A/B 分別）
        feat_A = feature_scaler_A.transform(df_t[feature_cols])
        feat_B = feature_scaler_B.transform(df_t[feature_cols])

        # LSTM_A 輸入
        df_t_lstmA = df_t.copy()
        df_t_lstmA[feature_cols] = feat_A

        # 建 LSTM 測試序列並預測 Δv
        X_test, _, _ = create_lstm_data(df_t_lstmA, feature_cols, window=WINDOW, shift=PREDICT_SHIFT)
        lstm_pred_dv = lstm_A.predict(X_test, verbose=0).flatten()
        base_speed = df_t_lstmA["TravelSpeed"].values[WINDOW-1:WINDOW-1+len(lstm_pred_dv)]
        lstm_pred_real = scaler_lstm.inverse_transform((lstm_pred_dv + base_speed).reshape(-1, 1)).flatten()

        # 回填預測速度
        df_t["lstm_pred_speed"] = np.nan
        start_idx = WINDOW
        df_t.iloc[start_idx:start_idx + len(lstm_pred_real), df_t.columns.get_loc("lstm_pred_speed")] = lstm_pred_real

        # 驗證資料（需有 lstm_pred_speed）
        df_valid = df_t.dropna(subset=["lstm_pred_speed"]).copy()
        y_true = df_valid["target_warning"].values
        is_peak = df_valid["is_peak"].values.astype(bool)

        pos_valid = df_t.index.get_indexer(df_valid.index)
        X_valid_A = np.hstack([feat_A[pos_valid, :], df_valid[["lstm_pred_speed", "總車輛數", "is_peak"]].values])
        X_valid_B = np.hstack([feat_B[pos_valid, :], df_valid[["lstm_pred_speed", "總車輛數", "is_peak"]].values])

        y_pred_A = xgb_A.predict(X_valid_A)
        y_pred_B = xgb_B.predict(X_valid_B)
        y_pred = np.where(is_peak, y_pred_B, y_pred_A)

        f1  = f1_score(y_true, y_pred, zero_division=0)
        mae = mean_absolute_error(df_valid["TravelSpeed_real"], df_valid["lstm_pred_speed"])
        rmse = np.sqrt(mean_squared_error(df_valid["TravelSpeed_real"], df_valid["lstm_pred_speed"]))

        plot_confusion(
            y_true, y_pred,
            os.path.join(output_dir, f"ensemble_confusion_{rid}_B{b_id}_final2025.png"),
            rid, b_id
        )

        summary_rows.append({"路段": rid, "Block": b_id, "F1": f1, "MAE": mae, "RMSE": rmse})
        block_scores.append((f1, mae, rmse))
        print(f"   ✅ {rid}-B{b_id} | Final(2025) F1={f1:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")
        print("   💾 已存：LSTM/XGB/Scaler（附 block 編號）")

    # 路段彙整（5 個 block → 平均 ± SE；四捨五入至 1 位）
    arr = np.array(block_scores)
    means = arr.mean(axis=0)
    ses   = arr.std(axis=0, ddof=1) / np.sqrt(len(arr))

    def fmt(m, se):
        return f"{m:.1f} ± {se:.1f}"

    agg_rows.append({
        "路段": rid,
        "F1(平均±SE)":   fmt(means[0], ses[0]),
        "MAE(平均±SE)":  fmt(means[1], ses[1]),
        "RMSE(平均±SE)": fmt(means[2], ses[2]),
    })
    print(f"\n📌 路段 {rid} Final(2025)：F1 {fmt(means[0], ses[0])} | MAE {fmt(means[1], ses[1])} | RMSE {fmt(means[2], ses[2])}")

# ===== 匯出成績 =====
pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "metrics_ensemble_blocks_detail_final2025.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(agg_rows).to_csv(os.path.join(output_dir, "metrics_ensemble_blocks_summary_final2025.csv"), index=False, encoding="utf-8-sig")

print("\n🎉 完成！")
print("   -> 每路段×每Block 明細：metrics_ensemble_blocks_detail_final2025.csv")
print("   -> 每路段平均 ± 標準誤（最終 2025 測試）：metrics_ensemble_blocks_summary_final2025.csv")
print("   -> 混淆矩陣：ensemble_confusion_<rid>_B<b>_final2025.png")
print(f"   -> Block 定義檔：{os.path.join(blocks_dir, 'blocks_<rid>.json')}")
