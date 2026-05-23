# analysis.py — 歷史股價分析

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 解決中文顯示問題（Windows）
plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

# ── 1. 抓取一年資料 ──────────────────────────
print("下載資料中...")
df = yf.download("2330.TW", period="1y", progress=False)
df = df[["Close", "Volume"]].copy()
df.columns = ["close", "volume"]
df.index = pd.to_datetime(df.index)

# ── 2. 計算技術指標 ───────────────────────────
df["ma20"]  = df["close"].rolling(20).mean()   # 20日均線
df["ma60"]  = df["close"].rolling(60).mean()   # 60日均線
df["daily_return"] = df["close"].pct_change()  # 日報酬率
df["cum_return"]   = (1 + df["daily_return"]).cumprod() - 1  # 累積報酬率

# ── 3. 印出基本統計 ───────────────────────────
print("\n── 台積電近一年統計 ──")
print(f"最高收盤：{df['close'].max():.1f} 元")
print(f"最低收盤：{df['close'].min():.1f} 元")
print(f"今日收盤：{df['close'].iloc[-1]:.1f} 元")
print(f"年化報酬：{df['cum_return'].iloc[-1]*100:+.2f}%")
print(f"日報酬標準差（波動率）：{df['daily_return'].std()*100:.2f}%")

# ── 4. 畫圖（3張子圖）────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 9),
                          gridspec_kw={"height_ratios": [3, 1, 1]})
fig.suptitle("台積電（2330）近一年分析", fontsize=14, fontweight="bold")

# 子圖一：股價 + 均線
ax1 = axes[0]
ax1.plot(df.index, df["close"], color="#2563EB", linewidth=1.2, label="收盤價")
ax1.plot(df.index, df["ma20"],  color="#F59E0B", linewidth=1,   label="MA20", linestyle="--")
ax1.plot(df.index, df["ma60"],  color="#EF4444", linewidth=1,   label="MA60", linestyle="--")
ax1.set_ylabel("股價（元）")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(axis="y", alpha=0.3)

# 子圖二：成交量
ax2 = axes[1]
colors = ["#16A34A" if r >= 0 else "#DC2626"
          for r in df["daily_return"].fillna(0)]
ax2.bar(df.index, df["volume"], color=colors, alpha=0.7, width=1)
ax2.set_ylabel("成交量")
ax2.grid(axis="y", alpha=0.3)

# 子圖三：累積報酬率
ax3 = axes[2]
ax3.plot(df.index, df["cum_return"] * 100,
         color="#7C3AED", linewidth=1.2)
ax3.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax3.fill_between(df.index, df["cum_return"] * 100, 0,
                 where=df["cum_return"] >= 0,
                 alpha=0.15, color="#7C3AED")
ax3.set_ylabel("累積報酬（%）")
ax3.grid(axis="y", alpha=0.3)

# 共用 X 軸格式
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

plt.tight_layout()
import os  # 如果 analysis.py 最上方沒有 import os，請記得補上這行

# === 自動取得程式碼所在的資料夾，並把圖片存進去 ===
# 1. 取得目前 analysis.py 所在的資料夾路徑
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 組合路徑與圖片名稱
filename_png = os.path.join(current_dir, "tsmc_analysis.png")

# 3. 儲存圖片
plt.savefig(filename_png, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n圖表已存檔：{filename_png}")