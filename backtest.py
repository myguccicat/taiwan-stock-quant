# backtest.py — 均線交叉策略回測

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

# ── 1. 下載資料 ───────────────────────────────
print("下載資料...")
df = yf.download("2330.TW", period="1y", progress=False)
df = df[["Close"]].copy()
df.columns = ["close"]

# ── 2. 計算均線與訊號 ─────────────────────────
df["ma20"] = df["close"].rolling(20).mean()
df["ma60"] = df["close"].rolling(60).mean()

# 訊號：ma20 在 ma60 之上 → 持有(1)，否則 → 空手(0)
df["signal"] = 0
df.loc[df["ma20"] > df["ma60"], "signal"] = 1

# 找出買進點和賣出點（訊號改變的那天）
df["position"] = df["signal"].diff()
buy_dates  = df[df["position"] ==  1].index
sell_dates = df[df["position"] == -1].index

print(f"\n共觸發買進訊號：{len(buy_dates)} 次")
print(f"共觸發賣出訊號：{len(sell_dates)} 次")

# ── 3. 計算策略報酬 ───────────────────────────
df["daily_return"]    = df["close"].pct_change()
df["strategy_return"] = df["daily_return"] * df["signal"].shift(1)

df["cum_market"]   = (1 + df["daily_return"]).cumprod() - 1
df["cum_strategy"] = (1 + df["strategy_return"]).cumprod() - 1

# ── 4. 績效指標 ───────────────────────────────
total_return  = df["cum_strategy"].iloc[-1] * 100
market_return = df["cum_market"].iloc[-1] * 100

daily_std = df["strategy_return"].std()
sharpe    = (df["strategy_return"].mean() / daily_std * (252 ** 0.5)
             if daily_std > 0 else 0)

rolling_max  = df["cum_strategy"].cummax()
max_drawdown = ((df["cum_strategy"] - rolling_max) / (1 + rolling_max)).min() * 100

wins  = (df["strategy_return"] > 0).sum()
total = (df["strategy_return"] != 0).sum()
win_rate = wins / total * 100 if total > 0 else 0

print("\n── 策略績效報告 ──────────────────")
print(f"策略累積報酬：{total_return:+.2f}%")
print(f"買入持有報酬：{market_return:+.2f}%")
print(f"Sharpe Ratio ：{sharpe:.2f}")
print(f"最大回撤     ：{max_drawdown:.2f}%")
print(f"勝率         ：{win_rate:.1f}%")
print("─────────────────────────────────")

# ── 5. 畫圖 ──────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle("台積電均線交叉策略回測", fontsize=13, fontweight="bold")

# 子圖一：股價 + 均線 + 買賣點
ax1 = axes[0]
ax1.plot(df.index, df["close"], color="#94A3B8", linewidth=1,   label="收盤價", alpha=0.8)
ax1.plot(df.index, df["ma20"],  color="#F59E0B", linewidth=1.2, label="MA20", linestyle="--")
ax1.plot(df.index, df["ma60"],  color="#EF4444", linewidth=1.2, label="MA60", linestyle="--")

# 標出買賣點
ax1.scatter(buy_dates,  df.loc[buy_dates,  "close"],
            marker="^", color="#16A34A", s=80, zorder=5, label="買進")
ax1.scatter(sell_dates, df.loc[sell_dates, "close"],
            marker="v", color="#DC2626", s=80, zorder=5, label="賣出")

ax1.set_ylabel("股價（元）")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(axis="y", alpha=0.3)

# 子圖二：累積報酬對比
ax2 = axes[1]
ax2.plot(df.index, df["cum_market"]   * 100,
         color="#94A3B8", linewidth=1.2, label="買入持有", linestyle="--")
ax2.plot(df.index, df["cum_strategy"] * 100,
         color="#7C3AED", linewidth=1.5, label="均線策略")
ax2.axhline(0, color="gray", linewidth=0.8, linestyle=":")
ax2.fill_between(df.index, df["cum_strategy"] * 100, 0,
                 where=df["cum_strategy"] >= 0,
                 alpha=0.1, color="#7C3AED")
ax2.set_ylabel("累積報酬（%）")
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
import os  # 如果程式碼最上方沒有 import os，記得補上這行

# === 自動取得程式碼所在的資料夾，並把圖片存進去 ===
# 1. 取得目前 backtest.py 所在的資料夾路徑
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 組合路徑與圖片名稱
filename_png = os.path.join(current_dir, "backtest_result.png")

# 3. 儲存圖片
plt.savefig(filename_png, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n圖表已存檔：{filename_png}")