import pandas as pd

def generate_flow_limit_table(excel_file, output_csv="flow_limit_table.csv"):
    df = pd.read_excel(excel_file)
    df["時間點"] = pd.to_datetime(df["時間點"])
    df["weekday"] = df["時間點"].dt.weekday
    df = df[df["weekday"] < 5]  # 僅平日

    df["時間字串"] = df["時間點"].dt.strftime("%H:%M")

    # 計算每 5 分鐘時間的 95% 分位數
    limit_table = df.groupby("時間字串")["總車輛數"].quantile(0.95).reset_index()
    limit_table.columns = ["時間字串", "流量上限"]
    limit_table.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 流量上限表已儲存至 {output_csv}")
