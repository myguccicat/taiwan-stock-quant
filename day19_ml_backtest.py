# day19_ml_backtest.py — ML 訊號回測系統

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from watchlist import ALL_STOCKS
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

# ══════════════════════════════════════════
#  設定區
# ══════════════════════════════════════════
PERIOD    = "3y"
TOP_N     = 5       # 每次做多幾檔
REBAL     = 5       # 幾天重新換倉一次（對應預測的 5 日報酬）
COST_RATE = 0.001425 + 0.003  # 手續費 0.1425% + 證交稅 0.3%
# ══════════════════════════════════════════

def calc_rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l)

def build_features(price, volume):
    df = pd.DataFrame(index=price.index)
    r1 = price.pct_change(1)
    df["r1"]         = r1
    df["r5"]         = price.pct_change(5)
    df["r20"]        = price.pct_change(20)
    df["ma5_ratio"]  = price / price.rolling(5).mean()  - 1
    df["ma20_ratio"] = price / price.rolling(20).mean() - 1
    df["ma60_ratio"] = price / price.rolling(60).mean() - 1
    df["vol_ratio"]  = volume / volume.rolling(20).mean()
    df["vol_5d"]     = r1.rolling(5).std()
    df["rsi14"]      = calc_rsi(price)
    bb = price.rolling(20).mean()
    bs = price.rolling(20).std()
    df["bb_pos"]     = (price - (bb - 2*bs)) / (4*bs + 1e-9)
    df["near_high"]  = price / price.rolling(20).max() - 1
    df["future_5d"]  = price.pct_change(5).shift(-5)
    return df

FEATURES = ["r1","r5","r20","ma5_ratio","ma20_ratio","ma60_ratio",
            "vol_ratio","vol_5d","rsi14","bb_pos","near_high"]

# ── 1. 下載資料 ───────────────────────────────
print(f"下載 {len(ALL_STOCKS)} 檔資料...")
raw     = yf.download(list(ALL_STOCKS.keys()), period=PERIOD,
                      progress=False, auto_adjust=True)
prices  = raw["Close"].rename(columns=ALL_STOCKS).ffill().dropna(axis=1, thresh=200)
volumes = raw["Volume"].rename(columns=ALL_STOCKS).ffill()
prices, volumes = prices.align(volumes[prices.columns], join="inner")
names   = list(prices.columns)
print(f"可用股票：{len(names)} 檔，{len(prices)} 個交易日")

# 下載大盤過濾用資料（台灣加權指數）
print("下載大盤資料...")
twii = yf.download("^TWII", period=PERIOD, progress=False, auto_adjust=True)["Close"]
twii_ma20 = twii.rolling(20).mean()
# 大盤過濾條件：加權指數在 MA20 之上才允許做多
def market_ok(date):
    try:
        idx = twii.index.get_indexer([date], method="ffill")[0]
        if idx < 0: return True
        return float(twii.iloc[idx]) > float(twii_ma20.iloc[idx])
    except:
        return True

# ── 2. 建立特徵面板 ───────────────────────────
print("建立特徵資料集...")
frames = []
for name in names:
    f = build_features(prices[name], volumes[name])
    f["stock"] = name
    frames.append(f)
panel = pd.concat(frames).dropna()

# ── 3. 時間切割 ───────────────────────────────
dates      = sorted(panel.index.unique())
split_date = dates[int(len(dates) * 0.7)]   # 前70%訓練，後30%測試
train = panel[panel.index <= split_date]
test  = panel[panel.index >  split_date]
print(f"訓練：{dates[0].date()} ～ {split_date.date()}")
print(f"測試：{split_date.date()} ～ {dates[-1].date()}")

# ── 4. 訓練模型 ───────────────────────────────
print("訓練模型...")
model = RandomForestRegressor(
    n_estimators=300, max_depth=6,
    min_samples_leaf=20, random_state=42, n_jobs=-1
)
model.fit(train[FEATURES], train["future_5d"])

# ── 5. 每日預測排名 ───────────────────────────
test = test.copy()
test["pred"] = model.predict(test[FEATURES])

# ── 6. 模擬換倉邏輯 ───────────────────────────
# 每 REBAL 天換倉一次，做多預測分數前 TOP_N 名
rebal_dates = [d for i, d in enumerate(sorted(test.index.unique())) if i % REBAL == 0]

portfolio_returns = []
prev_holdings     = []

for i, date in enumerate(rebal_dates[:-1]):
    next_date = rebal_dates[i + 1]

    # 大盤過濾：指數低於 MA20 時空手
    if not market_ok(date):
        selected = []  # 空手
        avg_ret  = 0.0
        cost     = 0.0
        net_ret  = 0.0
        portfolio_returns.append({
            "date": date, "return": net_ret,
            "gross": avg_ret, "cost": cost,
            "holdings": [], "market_filter": True,
        })
        prev_holdings = []
        continue

    # 當天選股
    day_data = test.loc[date].sort_values("pred", ascending=False)
    selected = day_data["stock"].iloc[:TOP_N].tolist()

    # 計算持有到下次換倉的報酬
    period_rets = []
    for stock in selected:
        try:
            p = prices[stock]
            ret = (p.loc[next_date] - p.loc[date]) / p.loc[date]
            period_rets.append(ret)
        except:
            period_rets.append(0.0)

    avg_ret = np.mean(period_rets)

    # 扣除換倉成本
    turnover = len(set(selected) - set(prev_holdings)) / TOP_N
    cost     = turnover * COST_RATE
    net_ret  = avg_ret - cost

    portfolio_returns.append({
        "date":     date,
        "return":   net_ret,
        "gross":    avg_ret,
        "cost":     cost,
        "holdings": selected,
        "market_filter": False,
    })
    prev_holdings = selected

res = pd.DataFrame(portfolio_returns).set_index("date")

# ── 7. 計算基準（等權持有所有股票）────────────
bm_rets = []
for i, date in enumerate(rebal_dates[:-1]):
    next_date = rebal_dates[i + 1]
    day_rets  = []
    for stock in names:
        try:
            ret = (prices[stock].loc[next_date] - prices[stock].loc[date]) / prices[stock].loc[date]
            day_rets.append(ret)
        except:
            pass
    bm_rets.append({"date": date, "return": np.mean(day_rets)})

bm = pd.DataFrame(bm_rets).set_index("date")

# ── 8. 績效指標 ───────────────────────────────
ml_cum  = (1 + res["return"]).cumprod() - 1
bm_cum  = (1 + bm["return"]).cumprod() - 1

ml_annual  = (1 + res["return"].mean()) ** (252 / REBAL) - 1
bm_annual  = (1 + bm["return"].mean()) ** (252 / REBAL) - 1
ml_sharpe  = res["return"].mean() / res["return"].std() * np.sqrt(252 / REBAL)
ml_maxdd   = ((ml_cum + 1) / (ml_cum + 1).cummax() - 1).min()
win_rate   = (res["return"] > 0).mean()
beat_bm    = (res["return"] > bm["return"]).mean()

filter_days = res["market_filter"].sum() if "market_filter" in res.columns else 0
print(f"║ 大盤過濾空手次數：{filter_days:>4} 次")
print(f"""
╔══════════════════════════════════════════╗
║        ML 選股策略回測結果               ║
╠══════════════════════════════════════════╣
║ 測試期間換倉次數：{len(res):>4} 次
║ ML策略年化報酬  ：{ml_annual*100:>+7.1f}%
║ 基準年化報酬    ：{bm_annual*100:>+7.1f}%
║ 超額年化報酬    ：{(ml_annual-bm_annual)*100:>+7.1f}%
║ Sharpe Ratio  ：{ml_sharpe:>7.2f}
║ 最大回撤       ：{ml_maxdd*100:>7.1f}%
║ 勝率           ：{win_rate*100:>7.1f}%
║ 跑贏基準比例   ：{beat_bm*100:>7.1f}%
║ 總交易成本     ：{res['cost'].sum()*100:>7.2f}%
╚══════════════════════════════════════════╝
""")

# 最近5次換倉選股紀錄
print("── 最近 5 次換倉選股 ──")
for date, row in res.tail(5).iterrows():
    print(f"  {date.date()}  {', '.join(row['holdings'])}  "
          f"報酬:{row['return']*100:+.2f}%")

# ── 9. 畫圖 ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f"ML 選股策略回測（Top{TOP_N}，每{REBAL}日換倉，含交易成本）",
             fontsize=13, fontweight="bold")

# 左上：累積報酬對比
ax1 = axes[0, 0]
ax1.plot(ml_cum.index, ml_cum * 100,
         color="#7C3AED", lw=1.8, label=f"ML策略 {ml_cum.iloc[-1]*100:+.1f}%")
ax1.plot(bm_cum.index, bm_cum * 100,
         color="#94A3B8", lw=1.2, ls="--",
         label=f"等權基準 {bm_cum.iloc[-1]*100:+.1f}%")
ax1.axhline(0, color="gray", lw=0.5, ls=":")
ax1.fill_between(ml_cum.index,
                 ml_cum * 100, bm_cum.reindex(ml_cum.index).ffill() * 100,
                 where=ml_cum >= bm_cum.reindex(ml_cum.index).ffill(),
                 alpha=0.1, color="#7C3AED", label="跑贏區間")
ax1.set_title("ML策略 vs 等權基準")
ax1.set_ylabel("累積報酬（%）")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# 右上：每次換倉報酬分佈
ax2 = axes[0, 1]
ax2.hist(res["return"] * 100, bins=30,
         color="#7C3AED", alpha=0.7, label="ML策略")
ax2.hist(bm["return"] * 100, bins=30,
         color="#94A3B8", alpha=0.5, label="基準")
ax2.axvline(res["return"].mean() * 100,
            color="#EF4444", lw=1.5, ls="--",
            label=f"ML均值 {res['return'].mean()*100:+.2f}%")
ax2.axvline(0, color="gray", lw=0.8)
ax2.set_title(f"每次換倉報酬分佈（{REBAL}日）")
ax2.set_xlabel("報酬率（%）")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# 左下：回撤曲線
ax3 = axes[1, 0]
drawdown = (ml_cum + 1) / (ml_cum + 1).cummax() - 1
ax3.fill_between(drawdown.index, drawdown * 100, 0,
                 alpha=0.4, color="#EF4444")
ax3.plot(drawdown.index, drawdown * 100,
         color="#DC2626", lw=1.2)
ax3.set_title(f"回撤曲線（最大回撤 {ml_maxdd*100:.1f}%）")
ax3.set_ylabel("回撤（%）")
ax3.grid(alpha=0.3)

# 右下：累積超額報酬（alpha）
ax4 = axes[1, 1]
alpha_cum = ml_cum - bm_cum.reindex(ml_cum.index).ffill()
ax4.plot(alpha_cum.index, alpha_cum * 100,
         color="#1D9E75", lw=1.5)
ax4.axhline(0, color="gray", lw=0.8, ls="--")
ax4.fill_between(alpha_cum.index, alpha_cum * 100, 0,
                 where=alpha_cum >= 0,
                 alpha=0.15, color="#1D9E75")
ax4.fill_between(alpha_cum.index, alpha_cum * 100, 0,
                 where=alpha_cum < 0,
                 alpha=0.15, color="#EF4444")
ax4.set_title("累積超額報酬（Alpha）")
ax4.set_ylabel("vs 基準（%）")
ax4.grid(alpha=0.3)

plt.tight_layout()
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "day19_ml_backtest.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n圖表已存檔：{out}")