# day11_multi_backtest.py — 多股票均線策略回測

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

# ── 1. 股票清單 ───────────────────────────────
STOCKS = {
    "2330.TW": "台積電", "2317.TW": "鴻海",
    "2454.TW": "聯發科", "2308.TW": "台達電",
    "2382.TW": "廣達",   "2303.TW": "聯電",
    "2412.TW": "中華電", "2882.TW": "國泰金",
    "1301.TW": "台塑",   "2357.TW": "華碩",
}

# ── 2. 回測單一股票的函式 ─────────────────────
def backtest_one(prices: pd.Series, ma_short=20, ma_long=60) -> dict:
    df = pd.DataFrame({"close": prices})
    df["ma_s"] = df["close"].rolling(ma_short).mean()
    df["ma_l"] = df["close"].rolling(ma_long).mean()
    df["signal"]   = (df["ma_s"] > df["ma_l"]).astype(int)
    df["ret"]      = df["close"].pct_change()
    df["strat_ret"] = df["ret"] * df["signal"].shift(1)
    df = df.dropna()

    cum = (1 + df["strat_ret"]).cumprod()
    mkt = (1 + df["ret"]).cumprod()

    total_ret   = cum.iloc[-1] - 1
    market_ret  = mkt.iloc[-1] - 1
    std         = df["strat_ret"].std()
    sharpe      = df["strat_ret"].mean() / std * np.sqrt(252) if std > 0 else 0
    roll_max    = cum.cummax()
    max_dd      = ((cum - roll_max) / roll_max).min()
    wins        = (df["strat_ret"] > 0).sum()
    total_days  = (df["strat_ret"] != 0).sum()
    win_rate    = wins / total_days if total_days > 0 else 0

    return {
        "策略報酬":   round(total_ret  * 100, 1),
        "市場報酬":   round(market_ret * 100, 1),
        "超額報酬":   round((total_ret - market_ret) * 100, 1),
        "Sharpe":     round(sharpe, 2),
        "最大回撤":   round(max_dd * 100, 1),
        "勝率":       round(win_rate * 100, 1),
        "_cum":       cum,
        "_mkt":       mkt,
    }

# ── 3. 跑所有股票 ─────────────────────────────
print("下載資料...")
raw = yf.download(list(STOCKS.keys()), period="1y", progress=False)["Close"]
raw.columns = list(STOCKS.values())
raw = raw.dropna()

print("\n回測中...\n")
results = {}
for name in raw.columns:
    results[name] = backtest_one(raw[name])

# ── 4. 印出績效排行 ───────────────────────────
cols = ["策略報酬", "市場報酬", "超額報酬", "Sharpe", "最大回撤", "勝率"]
df_result = pd.DataFrame(
    {k: {c: v[c] for c in cols} for k, v in results.items()}
).T.sort_values("Sharpe", ascending=False)

print("── 多股票回測績效排行（依 Sharpe 排序）──")
print(f"{'股票':<6} {'策略%':>7} {'市場%':>7} {'超額%':>7} "
      f"{'Sharpe':>7} {'回撤%':>7} {'勝率%':>7}")
print("─" * 55)
for name, row in df_result.iterrows():
    exceed_mark = "▲" if row["超額報酬"] > 0 else "▼"
    print(f"{name:<6} {row['策略報酬']:>7.1f} {row['市場報酬']:>7.1f} "
          f"{row['超額報酬']:>6.1f}{exceed_mark} {row['Sharpe']:>7.2f} "
          f"{row['最大回撤']:>7.1f} {row['勝率']:>7.1f}")

beat_market = (df_result["超額報酬"] > 0).sum()
print(f"\n策略跑贏市場：{beat_market} / {len(STOCKS)} 檔")
print(f"平均 Sharpe：{df_result['Sharpe'].mean():.2f}")
print(f"平均最大回撤：{df_result['最大回撤'].mean():.1f}%")

# ── 5. 畫圖 ──────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle("10 檔股票均線策略回測——淨值曲線對比",
             fontsize=13, fontweight="bold")

for idx, (name, res) in enumerate(results.items()):
    ax = axes[idx // 5][idx % 5]
    cum = res["_cum"] * 100 - 100
    mkt = res["_mkt"] * 100 - 100

    ax.plot(cum.index, cum.values,
            color="#7C3AED", linewidth=1.2, label="策略")
    ax.plot(mkt.index, mkt.values,
            color="#94A3B8", linewidth=1,
            linestyle="--", label="持有", alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.fill_between(cum.index, cum.values, 0,
                    where=cum.values >= 0,
                    alpha=0.08, color="#7C3AED")

    color = "#16A34A" if res["超額報酬"] > 0 else "#DC2626"
    ax.set_title(
        f"{name}  Sharpe:{res['Sharpe']}",
        fontsize=9, fontweight="500"
    )
    ax.set_ylabel("累積報酬（%）", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter("%m/%d")
    )
    ax.xaxis.set_major_locator(
        plt.matplotlib.dates.MonthLocator(interval=2)
    )
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(alpha=0.3)

plt.tight_layout()


import os  # 請確認您的檔案最上方有這行，如果沒有請補上

# 1. 自動抓取目前這份程式碼檔案的「絕對路徑資料夾」
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 把資料夾路徑與檔名組合在一起
save_path = os.path.join(current_dir, "day11_multi_backtest.png")

# 3. 儲存與顯示
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n圖表已存檔：{save_path}")