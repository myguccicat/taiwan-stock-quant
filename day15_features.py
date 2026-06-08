# day15_features.py — 特徵工程：建立 ML 訓練資料集

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

# ── 1. 下載資料 ───────────────────────────────
print("下載資料...")
df = yf.download("2330.TW", period="2y", progress=False)
df = df[["Open","High","Low","Close","Volume"]].copy()
df.columns = ["open","high","low","close","volume"]

# ── 2. 建立技術指標特徵 ───────────────────────
# 價格類
df["return_1d"]  = df["close"].pct_change(1)   # 昨日報酬率
df["return_5d"]  = df["close"].pct_change(5)   # 5日報酬率
df["return_20d"] = df["close"].pct_change(20)  # 20日報酬率

# 均線類
df["ma5"]  = df["close"].rolling(5).mean()
df["ma20"] = df["close"].rolling(20).mean()
df["ma60"] = df["close"].rolling(60).mean()

# 均線比值（比絕對數字更有意義）
df["ma5_ratio"]  = df["close"] / df["ma5"]  - 1   # 離 MA5 的距離
df["ma20_ratio"] = df["close"] / df["ma20"] - 1
df["ma60_ratio"] = df["close"] / df["ma60"] - 1
df["ma5_20_cross"] = df["ma5"] / df["ma20"] - 1   # 均線交叉強度

# 波動率類
df["vol_5d"]  = df["return_1d"].rolling(5).std()   # 5日波動率
df["vol_20d"] = df["return_1d"].rolling(20).std()  # 20日波動率

# 成交量類
df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()  # 量比

# RSI（相對強弱指標）
def calc_rsi(prices, n=14):
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

df["rsi14"] = calc_rsi(df["close"], 14)

# 布林通道位置（0=下緣, 1=上緣）
bb_mid = df["close"].rolling(20).mean()
bb_std = df["close"].rolling(20).std()
df["bb_position"] = (df["close"] - (bb_mid - 2*bb_std)) / (4 * bb_std)

# 高低點突破
df["high_20d"] = df["high"].rolling(20).max()
df["low_20d"]  = df["low"].rolling(20).min()
df["near_high"] = df["close"] / df["high_20d"] - 1  # 距20日高點距離
df["near_low"]  = df["close"] / df["low_20d"]  - 1  # 距20日低點距離

# ── 3. 建立預測目標（Label）────────────────────
# 目標：明天的報酬率是正還是負？
# 用「未來 5 日報酬率」預測，比單日穩定
df["future_5d_return"] = df["close"].pct_change(5).shift(-5)
df["label"] = (df["future_5d_return"] > 0).astype(int)
# 1 = 未來5天上漲, 0 = 下跌或持平

# ── 4. 整理最終資料集 ─────────────────────────
FEATURES = [
    "return_1d", "return_5d", "return_20d",
    "ma5_ratio", "ma20_ratio", "ma60_ratio", "ma5_20_cross",
    "vol_5d", "vol_20d", "vol_ratio",
    "rsi14", "bb_position",
    "near_high", "near_low",
]

dataset = df[FEATURES + ["label"]].dropna()
dataset.to_csv("ml_dataset.csv")

print(f"\n資料集建立完成！")
print(f"  總樣本數：{len(dataset)} 筆")
print(f"  特徵數量：{len(FEATURES)} 個")
print(f"  上漲樣本：{dataset['label'].sum()} 筆（{dataset['label'].mean()*100:.1f}%）")
print(f"  下跌樣本：{(dataset['label']==0).sum()} 筆（{(1-dataset['label'].mean())*100:.1f}%）")

print(f"\n── 特徵統計 ──")
print(dataset[FEATURES].describe().round(4).to_string())

# ── 5. 特徵相關性圖 ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("特徵工程分析", fontsize=13, fontweight="bold")

# 左：特徵和 label 的相關性
ax1 = axes[0]
corr_with_label = dataset[FEATURES].corrwith(dataset["label"]).sort_values()
colors = ["#16A34A" if v > 0 else "#DC2626" for v in corr_with_label]
ax1.barh(corr_with_label.index, corr_with_label.values, color=colors, alpha=0.8)
ax1.axvline(0, color="gray", lw=0.8)
ax1.set_title("各特徵與「未來5日漲跌」的相關性")
ax1.set_xlabel("相關係數")
ax1.grid(axis="x", alpha=0.3)

# 右：特徵之間的相關性 heatmap
ax2 = axes[1]
corr_matrix = dataset[FEATURES].corr()
im = ax2.imshow(corr_matrix.values, cmap="RdYlGn", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax2, fraction=0.046)
ax2.set_xticks(range(len(FEATURES)))
ax2.set_yticks(range(len(FEATURES)))
ax2.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=7)
ax2.set_yticklabels(FEATURES, fontsize=7)
ax2.set_title("特徵間相關性（避免高度共線）")

plt.tight_layout()

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(current_dir, "day15_features.png")
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.show()
print("\n圖表已存檔：day15_features.png")