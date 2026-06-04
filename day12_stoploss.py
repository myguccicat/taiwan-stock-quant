# day12_stoploss.py — 含停損的多股票回測

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

STOCKS = {
    "2330.TW": "台積電", "2317.TW": "鴻海",
    "2454.TW": "聯發科", "2308.TW": "台達電",
    "2382.TW": "廣達",   "2303.TW": "聯電",
    "2412.TW": "中華電", "2882.TW": "國泰金",
    "1301.TW": "台塑",   "2357.TW": "華碩",
}

def backtest_with_stoploss(prices: pd.Series,
                            ma_short=20, ma_long=60,
                            stop_loss=-0.07) -> dict:
    """
    均線策略 + 停損：
    持倉期間若從進場價跌超過 stop_loss，立即出場，
    等下一個均線買進訊號再重新進場。
    """
    df = pd.DataFrame({"close": prices.copy()})
    df["ma_s"] = df["close"].rolling(ma_short).mean()
    df["ma_l"] = df["close"].rolling(ma_long).mean()
    df["signal"] = (df["ma_s"] > df["ma_l"]).astype(int)
    df["ret"] = df["close"].pct_change()
    df = df.dropna().copy()

    # 逐日模擬：加入停損邏輯
    position    = 0       # 0=空手, 1=持倉
    entry_price = 0.0
    strat_rets  = []

    for i in range(len(df)):
        row         = df.iloc[i]
        daily_ret   = row["ret"]
        close_price = row["close"]
        ma_signal   = row["signal"]

        if position == 1:
            # 計算從進場到今天的累積虧損
            loss = (close_price - entry_price) / entry_price
            if loss <= stop_loss:
                # 觸發停損，今日以停損價出場
                strat_rets.append(stop_loss)
                position = 0
                continue

        if position == 0 and ma_signal == 1:
            # 均線訊號買進
            position    = 1
            entry_price = close_price
            strat_rets.append(daily_ret)
        elif position == 1 and ma_signal == 0:
            # 均線訊號賣出
            strat_rets.append(daily_ret)
            position = 0
        elif position == 1:
            strat_rets.append(daily_ret)
        else:
            strat_rets.append(0.0)

    df["strat_ret"] = strat_rets

    # 績效計算
    cum = (1 + df["strat_ret"]).cumprod()
    mkt = (1 + df["ret"]).cumprod()

    total_ret  = cum.iloc[-1] - 1
    market_ret = mkt.iloc[-1] - 1
    std        = df["strat_ret"].std()
    sharpe     = df["strat_ret"].mean() / std * np.sqrt(252) if std > 0 else 0
    roll_max   = cum.cummax()
    max_dd     = ((cum - roll_max) / roll_max).min()
    wins       = (df["strat_ret"] > 0).sum()
    total_days = (df["strat_ret"] != 0).sum()
    win_rate   = wins / total_days if total_days > 0 else 0

    return {
        "策略報酬":  round(total_ret  * 100, 1),
        "市場報酬":  round(market_ret * 100, 1),
        "超額報酬":  round((total_ret - market_ret) * 100, 1),
        "Sharpe":    round(sharpe, 2),
        "最大回撤":  round(max_dd * 100, 1),
        "勝率":      round(win_rate * 100, 1),
        "_cum":      cum,
        "_mkt":      mkt,
    }

# 下載資料
print("下載資料...")
raw = yf.download(list(STOCKS.keys()), period="1y", progress=False)["Close"]
raw.columns = list(STOCKS.values())
raw = raw.dropna()

# 跑兩組回測：有停損 vs 無停損
print("\n回測中...\n")
res_with    = {n: backtest_with_stoploss(raw[n], stop_loss=-0.07)
               for n in raw.columns}
res_without = {n: backtest_with_stoploss(raw[n], stop_loss=-9999)
               for n in raw.columns}

# 比較表
cols = ["策略報酬", "Sharpe", "最大回撤"]
print(f"{'股票':<6} {'有停損報酬':>9} {'無停損報酬':>9} "
      f"{'有停損回撤':>9} {'無停損回撤':>9} {'停損效果'}")
print("─" * 65)
for name in raw.columns:
    w  = res_with[name]
    wo = res_without[name]
    dd_better  = "✓回撤改善" if w["最大回撤"] > wo["最大回撤"] else "─"
    ret_change = w["策略報酬"] - wo["策略報酬"]
    print(f"{name:<6} {w['策略報酬']:>8.1f}% {wo['策略報酬']:>8.1f}%"
          f" {w['最大回撤']:>8.1f}% {wo['最大回撤']:>8.1f}%"
          f"  {dd_better}  報酬{ret_change:+.1f}%")

# 畫圖：選 4 檔最有代表性的來比較
show_stocks = ["台積電", "鴻海", "廣達", "聯發科"]
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("停損機制效果比較（紫=有停損，橘=無停損，灰=買入持有）",
             fontsize=12, fontweight="bold")

for idx, name in enumerate(show_stocks):
    ax  = axes[idx // 2][idx % 2]
    cum_w  = res_with[name]["_cum"]  * 100 - 100
    cum_wo = res_without[name]["_cum"] * 100 - 100
    mkt    = res_with[name]["_mkt"]  * 100 - 100

    ax.plot(cum_w.index,  cum_w.values,
            color="#7C3AED", linewidth=1.3, label="有停損")
    ax.plot(cum_wo.index, cum_wo.values,
            color="#F59E0B", linewidth=1.3,
            linestyle="--", label="無停損")
    ax.plot(mkt.index,    mkt.values,
            color="#94A3B8", linewidth=1,
            linestyle=":", label="買入持有", alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    w = res_with[name]
    ax.set_title(
        f"{name}  有停損 Sharpe:{w['Sharpe']}  "
        f"回撤:{w['最大回撤']}%",
        fontsize=9
    )
    ax.set_ylabel("累積報酬（%）", fontsize=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter("%m/%d")
    )
    ax.xaxis.set_major_locator(
        plt.matplotlib.dates.MonthLocator(interval=2)
    )
    ax.grid(alpha=0.3)

plt.tight_layout()
import os  # 請確認您的檔案最上方有這行，如果沒有請補上

# 1. 自動抓取目前這份程式碼檔案的「絕對路徑資料夾」
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 把資料夾路徑與檔名組合在一起
save_path = os.path.join(current_dir, "day12_stoploss.png")

# 3. 儲存與顯示
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n圖表已存檔：{save_path}")