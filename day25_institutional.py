# day25_institutional.py — 三大法人籌碼特徵整合

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FinMind.data import DataLoader
from datetime import date, timedelta
from watchlist import ALL_STOCKS
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

dl = DataLoader()

# ── 1. 定義抓取函式 ───────────────────────────
def get_institutional(stock_id, days=200):
    """抓取單一股票三大法人資料，整理成每日買超金額"""
    start = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        df = dl.taiwan_stock_institutional_investors(
            stock_id=stock_id,
            start_date=start,
        )
        if df.empty:
            return None

        # 計算買超張數（買-賣）
        df["net"] = df["buy"] - df["sell"]

        # 轉成寬格式
        pivot = df.pivot_table(
            index="date", columns="name", values="net", aggfunc="sum"
        )
        pivot.index = pd.to_datetime(pivot.index)

        # 統一欄位名稱
        col_map = {
            "Foreign_Investor":   "foreign_net",
            "Investment_Trust":   "trust_net",
            "Dealer_self":        "dealer_net",
            "Dealer_Hedging":     "dealer_hedge",
            "Foreign_Dealer_Self":"foreign_dealer",
        }
        pivot = pivot.rename(columns=col_map)

        # 只保留主要三欄
        for col in ["foreign_net","trust_net","dealer_net"]:
            if col not in pivot.columns:
                pivot[col] = 0

        result = pivot[["foreign_net","trust_net","dealer_net"]].fillna(0)

        # 計算滾動特徵
        result["foreign_5d"]  = result["foreign_net"].rolling(5).sum()
        result["trust_5d"]    = result["trust_net"].rolling(5).sum()
        result["dealer_5d"]   = result["dealer_net"].rolling(5).sum()
        result["total_net"]   = result["foreign_net"] + result["trust_net"] + result["dealer_net"]
        result["total_5d"]    = result["total_net"].rolling(5).sum()

        return result

    except Exception as e:
        print(f"  {stock_id} 抓取失敗：{e}")
        return None

# ── 2. 測試幾檔股票 ───────────────────────────
test_stocks = {
    "2330": "台積電",
    "2454": "聯發科",
    "2317": "鴻海",
    "3680": "家登",
    "6239": "力成",
}

print("抓取三大法人資料中（每檔約 2 秒）...\n")
results = {}
for code, name in test_stocks.items():
    print(f"  下載 {name}（{code}）...")
    data = get_institutional(code, days=200)
    if data is not None:
        results[name] = data
        latest = data.iloc[-1]
        print(f"    外資買超：{latest['foreign_net']:>12,.0f} 股")
        print(f"    投信買超：{latest['trust_net']:>12,.0f} 股")
        print(f"    自營買超：{latest['dealer_net']:>12,.0f} 股")
        print(f"    合計買超：{latest['total_net']:>12,.0f} 股")
        print()

# ── 3. 視覺化 ─────────────────────────────────
fig, axes = plt.subplots(len(results), 1,
                          figsize=(12, 3*len(results)))
if len(results) == 1:
    axes = [axes]

for ax, (name, data) in zip(axes, results.items()):
    colors = ["#16A34A" if v >= 0 else "#DC2626"
              for v in data["total_net"]]
    ax.bar(data.index, data["total_net"] / 1000,
           color=colors, alpha=0.7, width=1)
    ax.plot(data.index, data["total_5d"] / 1000,
            color="#7C3AED", lw=1.5, label="5日合計")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title(f"{name} 三大法人每日買超（千股）")
    ax.set_ylabel("千股")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("day25_institutional.png", dpi=150, bbox_inches="tight")
plt.show()
print("圖表已存檔：day25_institutional.png")