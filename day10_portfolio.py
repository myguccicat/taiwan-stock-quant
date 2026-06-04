# day10_portfolio.py — 投資組合理論與效率前緣

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False
np.random.seed(42)

# ── 1. 選 6 檔相關性較低的股票 ────────────────
STOCKS = {
    "2330.TW": "台積電",
    "2412.TW": "中華電",
    "1301.TW": "台塑",
    "2454.TW": "聯發科",
    "2882.TW": "國泰金",
    "2308.TW": "台達電",
}

print("下載資料...")
raw = yf.download(list(STOCKS.keys()), period="1y", progress=False)["Close"]
raw.columns = list(STOCKS.values())
returns = raw.pct_change().dropna()

# ── 2. 基本統計 ───────────────────────────────
annual_ret = returns.mean() * 252
annual_vol = returns.std() * np.sqrt(252)
cov_matrix = returns.cov() * 252

print("\n── 各股年化指標 ──")
for name in STOCKS.values():
    sr = annual_ret[name] / annual_vol[name]
    print(f"  {name:<5}  報酬:{annual_ret[name]*100:+6.1f}%  "
          f"波動:{annual_vol[name]*100:5.1f}%  Sharpe:{sr:.2f}")

# ── 3. 蒙地卡羅模擬 10000 個隨機組合 ─────────
N = 10000
n_assets = len(STOCKS)
results = np.zeros((3, N))

for i in range(N):
    w = np.random.dirichlet(np.ones(n_assets))  # 隨機權重，總和=1
    p_ret = np.dot(w, annual_ret)
    p_vol = np.sqrt(w @ cov_matrix.values @ w)
    sharpe = p_ret / p_vol
    results[0, i] = p_vol * 100
    results[1, i] = p_ret * 100
    results[2, i] = sharpe

# ── 4. 找出最佳組合 ───────────────────────────
# 最高 Sharpe（風險調整後最佳）
best_sharpe_idx = results[2].argmax()
# 最低波動
min_vol_idx = results[0].argmin()

def get_weights(idx):
    np.random.seed(42)
    for i in range(N):
        w = np.random.dirichlet(np.ones(n_assets))
        if i == idx:
            return w
    return None

w_sharpe = get_weights(best_sharpe_idx)
w_minvol = get_weights(min_vol_idx)

names = list(STOCKS.values())
print("\n── 最高 Sharpe 組合 ──")
print(f"  報酬:{results[1, best_sharpe_idx]:.1f}%  "
      f"波動:{results[0, best_sharpe_idx]:.1f}%  "
      f"Sharpe:{results[2, best_sharpe_idx]:.2f}")
for n, w in zip(names, w_sharpe):
    print(f"  {n}：{w*100:.1f}%")

print("\n── 最低波動組合 ──")
print(f"  報酬:{results[1, min_vol_idx]:.1f}%  "
      f"波動:{results[0, min_vol_idx]:.1f}%  "
      f"Sharpe:{results[2, min_vol_idx]:.2f}")
for n, w in zip(names, w_minvol):
    print(f"  {n}：{w*100:.1f}%")

# ── 5. 畫圖 ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("投資組合效率前緣分析", fontsize=13, fontweight="bold")

# 左圖：效率前緣散點圖
ax1 = axes[0]
sc = ax1.scatter(results[0], results[1],
                 c=results[2], cmap="RdYlGn",
                 s=3, alpha=0.4)
plt.colorbar(sc, ax=ax1, label="Sharpe Ratio")

# 標出最佳點
ax1.scatter(results[0, best_sharpe_idx], results[1, best_sharpe_idx],
            color="#7C3AED", s=150, zorder=10,
            marker="*", label=f"最高Sharpe {results[2,best_sharpe_idx]:.2f}")
ax1.scatter(results[0, min_vol_idx], results[1, min_vol_idx],
            color="#1D4ED8", s=150, zorder=10,
            marker="D", label="最低波動")

# 標出個別股票
for name in names:
    ax1.scatter(annual_vol[name]*100, annual_ret[name]*100,
                s=60, zorder=8, color="black", marker="^")
    ax1.annotate(name,
                 (annual_vol[name]*100, annual_ret[name]*100),
                 textcoords="offset points", xytext=(5, 4), fontsize=8)

ax1.set_xlabel("年化波動率（%）")
ax1.set_ylabel("年化報酬率（%）")
ax1.set_title("效率前緣（10,000 個隨機組合）")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# 右圖：兩個最佳組合的權重圓餅圖
ax2 = axes[1]
ax2.axis("off")

# 手動畫兩個圓餅
ax_left  = fig.add_axes([0.55, 0.15, 0.18, 0.65])
ax_right = fig.add_axes([0.77, 0.15, 0.18, 0.65])

colors = ["#7C3AED","#1D9E75","#BA7517","#EF4444","#0C447C","#16A34A"]

ax_left.pie(w_sharpe, labels=names, colors=colors,
            autopct="%1.0f%%", textprops={"fontsize": 7})
ax_left.set_title("最高Sharpe\n組合配置", fontsize=9)

ax_right.pie(w_minvol, labels=names, colors=colors,
             autopct="%1.0f%%", textprops={"fontsize": 7})
ax_right.set_title("最低波動\n組合配置", fontsize=9)

plt.tight_layout()

import os  # 請確認您的檔案最上方有這行，如果沒有請補上

# 1. 自動抓取目前這份程式碼檔案的「絕對路徑資料夾」
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 把資料夾路徑與檔名組合在一起
save_path = os.path.join(current_dir, "day10_portfolio.png")

# 3. 儲存與顯示
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n圖表已存檔：{save_path}")