# day9_correlation.py — 多股票相關性分析

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

# ── 1. 定義股票清單 ───────────────────────────
STOCKS = {
    "2330": "台積電", "2317": "鴻海",   "2454": "聯發科",
    "2382": "廣達",   "2308": "台達電", "2303": "聯電",
    "2412": "中華電", "2882": "國泰金", "2881": "富邦金",
    "1301": "台塑",   "1303": "南亞",   "2002": "中鋼",
    "2886": "兆豐金", "3711": "日月光", "2357": "華碩",
}

codes = [f"{c}.TW" for c in STOCKS.keys()]
names = list(STOCKS.values())

# ── 2. 下載資料 ───────────────────────────────
print("下載 15 檔股票資料（約需 10 秒）...")
raw = yf.download(codes, period="1y", progress=False)["Close"]
raw.columns = names
raw = raw.dropna()

print(f"取得 {len(raw)} 個交易日的資料\n")

# ── 3. 計算日報酬率 & 相關係數矩陣 ───────────
returns = raw.pct_change().dropna()
corr    = returns.corr()

# ── 4. 印出關鍵資訊 ───────────────────────────
print("── 相關係數最高的 5 對（最相似）──")
pairs = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        pairs.append((names[i], names[j], corr.iloc[i, j]))

pairs.sort(key=lambda x: x[2], reverse=True)
for a, b, c in pairs[:5]:
    print(f"  {a} ↔ {b}：{c:.3f}")

print("\n── 相關係數最低的 5 對（最分散）──")
for a, b, c in pairs[-5:]:
    print(f"  {a} ↔ {b}：{c:.3f}")

# 各股與台積電的相關性
print("\n── 各股與台積電的相關性 ──")
tsmc_corr = corr["台積電"].drop("台積電").sort_values(ascending=False)
for name, val in tsmc_corr.items():
    bar = "█" * int(abs(val) * 20)
    print(f"  {name:<5}：{val:+.3f}  {bar}")

# ── 5. 畫圖 ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("台股 15 大股票相關性分析（近一年）",
             fontsize=13, fontweight="bold")

# 左圖：相關係數 heatmap
ax1 = axes[0]
cmap = plt.cm.RdYlGn
im = ax1.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
ax1.set_xticks(range(len(names)))
ax1.set_yticks(range(len(names)))
ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
ax1.set_yticklabels(names, fontsize=9)
ax1.set_title("相關係數矩陣（綠=高相關，紅=低相關）")

for i in range(len(names)):
    for j in range(len(names)):
        val = corr.iloc[i, j]
        color = "white" if abs(val) > 0.6 else "black"
        ax1.text(j, i, f"{val:.2f}", ha="center", va="center",
                 fontsize=7, color=color)

# 右圖：各股年化報酬 vs 波動率（風險報酬圖）
ax2 = axes[1]
annual_return = returns.mean() * 252 * 100
annual_vol    = returns.std()  * np.sqrt(252) * 100

scatter_colors = plt.cm.RdYlGn(
    (annual_return - annual_return.min()) /
    (annual_return.max() - annual_return.min())
)

sc = ax2.scatter(annual_vol, annual_return,
                 c=annual_return, cmap="RdYlGn",
                 s=80, zorder=5)
plt.colorbar(sc, ax=ax2, label="年化報酬（%）")

for i, name in enumerate(names):
    ax2.annotate(name,
                 (annual_vol.iloc[i], annual_return.iloc[i]),
                 textcoords="offset points", xytext=(5, 4),
                 fontsize=8)

ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax2.set_xlabel("年化波動率（%）")
ax2.set_ylabel("年化報酬率（%）")
ax2.set_title("風險報酬圖（右上角 = 高報酬高風險）")
ax2.grid(alpha=0.3)

plt.tight_layout()
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
filename_png = os.path.join(current_dir, "day9_correlation.png")
    
plt.savefig(filename_png, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n圖表已存檔：{filename_png}")