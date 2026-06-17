import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from watchlist import ALL_STOCKS
import warnings
import os
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

# ══════════════════════════════════════════
#  設定區
# ══════════════════════════════════════════
PERIOD    = "3y"
TOP_N     = 5       # 每次做多幾檔
REBAL     = 3       # 幾天重新換倉一次（對應預測的 3 日報酬）
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
    df["future_3d"]  = price.pct_change(3).shift(-3)  
    return df

FEATURES = ["r1","r5","r20","ma5_ratio","ma20_ratio","ma60_ratio",
            "vol_ratio","vol_5d","rsi14","bb_pos","near_high"]

# ── 1. 下載資料 ───────────────────────────────
print(f"下載 {len(ALL_STOCKS)} 檔資料...")
raw     = yf.download(list(ALL_STOCKS.keys()), period=PERIOD, progress=False, auto_adjust=True)
prices  = raw["Close"].rename(columns=ALL_STOCKS).ffill().dropna(axis=1, thresh=200)
volumes = raw["Volume"].rename(columns=ALL_STOCKS).ffill()
prices, volumes = prices.align(volumes[prices.columns], join="inner")
names   = list(prices.columns)
print(f"可用股票：{len(names)} 檔，{len(prices)} 個交易日")

# ── 2. 建立特徵面板 ──────────────────────────
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
    n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42, n_jobs=-1
)
model.fit(train[FEATURES], train["future_3d"])

# ── 5. 每日預測排名 ───────────────────────────
test = test.copy()
test["pred"] = model.predict(test[FEATURES])

# ── 6. 模擬換倉邏輯（3日高效自適應優化版 — 移除過期大盤風控） ──
rebal_dates = [d for i, d in enumerate(sorted(test.index.unique())) if i % REBAL == 0]

portfolio_returns = []
prev_holdings_dict = {}  

for i, date in enumerate(rebal_dates[:-1]):
    next_date = rebal_dates[i + 1]

    # 💡 修正位置：純 ML 模式，不調用未定義的 market_ok
    market_filter_tag = False 

    day_data = test.loc[date].set_index("stock")["pred"]
    
    if not prev_holdings_dict:
        selected_series = day_data.sort_values(ascending=False).iloc[:TOP_N]
        selected = selected_series.index.tolist()
        prev_holdings_dict = selected_series.to_dict()
    else:
        current_pool = day_data.sort_values(ascending=False)
        all_candidates = current_pool.index.tolist()
        new_selected = []
        keep_holdings = []
        for stock in prev_holdings_dict.keys():
            if stock in current_pool.index:
                rank = all_candidates.index(stock)
                if rank < 10: 
                    keep_holdings.append(stock)
        
        new_selected.extend(keep_holdings)
        
        for stock in all_candidates:
            if len(new_selected) == TOP_N:
                break
            if stock not in new_selected:
                new_selected.append(stock)
                
        selected = new_selected
        prev_holdings_dict = current_pool.loc[selected].to_dict()

    period_rets = []
    for stock in selected:
        try:
            p = prices[stock]
            ret = (p.loc[next_date] - p.loc[date]) / p.loc[date]
            period_rets.append(ret)
        except:
            period_rets.append(0.0)

    avg_ret = np.mean(period_rets)

    prev_stocks = list(portfolio_returns[-1]["holdings"]) if portfolio_returns else []
    turnover = len(set(selected) - set(prev_stocks)) / TOP_N if prev_stocks else 1.0
    cost     = turnover * COST_RATE
    net_ret  = avg_ret - cost

    portfolio_returns.append({
        "date":          date,
        "return":        net_ret,
        "gross":         avg_ret,
        "cost":          cost,
        "holdings":      selected,
        "market_filter": market_filter_tag,
    })

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

# ── 8. 績效指標（精準無溢價年化修正） ─────────────────
bm_aligned = bm["return"].reindex(res.index).ffill().fillna(0.0)

ml_cum  = (1 + res["return"]).cumprod() - 1
bm_cum  = (1 + bm_aligned).cumprod() - 1

# 💡 修正：改用實際回測涵蓋的天數進行精準標準化年化計算 (避免 mean 複利放大漏洞)
total_days = (res.index[-1] - res.index[0]).days
annual_factor = 365.25 / total_days if total_days > 0 else 1.0

ml_annual  = (ml_cum.iloc[-1] + 1) ** annual_factor - 1
bm_annual  = (bm_cum.iloc[-1] + 1) ** annual_factor - 1

# Sharpe Ratio 修正為標準每日對齊對照轉換
ml_sharpe  = (res["return"].mean() / (res["return"].std() + 1e-9)) * np.sqrt(252 / REBAL)
ml_maxdd   = ((ml_cum + 1) / (ml_cum + 1).cummax() - 1).min()
win_rate   = (res["return"] > 0).mean()
beat_bm    = (res["return"] > bm_aligned).mean()

print(f"""
╔══════════════════════════════════════════╗
║        ML 選股策略回測結果 (修正版)       ║
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

print("── 最近 5 次換倉選股 ──")
for date, row in res.tail(5).iterrows():
    print(f"  {date.date()}  {', '.join(row['holdings'])}  報酬:{row['return']*100:+.2f}%")

# ── 9. 畫圖 ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f"ML 選股策略回測（Top{TOP_N}，每{REBAL}日換倉，含交易成本）", fontsize=13, fontweight="bold")

ax1 = axes[0, 0]
ax1.plot(ml_cum.index, ml_cum * 100, color="#7C3AED", lw=1.8, label=f"ML策略 {ml_cum.iloc[-1]*100:+.1f}%")
ax1.plot(bm_cum.index, bm_cum * 100, color="#94A3B8", lw=1.2, ls="--", label=f"等權基準 {bm_cum.iloc[-1]*100:+.1f}%")
ax1.axhline(0, color="gray", lw=0.5, ls=":")
ax1.fill_between(ml_cum.index, ml_cum * 100, bm_cum.reindex(ml_cum.index).ffill() * 100,
                 where=ml_cum >= bm_cum.reindex(ml_cum.index).ffill(), alpha=0.1, color="#7C3AED", label="跑贏區間")
ax1.set_title("ML策略 vs 等權基準")
ax1.set_ylabel("累積報酬（%）")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

ax2 = axes[0, 1]
ax2.hist(res["return"] * 100, bins=30, color="#7C3AED", alpha=0.7, label="ML策略")
ax2.hist(bm["return"] * 100, bins=30, color="#94A3B8", alpha=0.5, label="基準")
ax2.axvline(res["return"].mean() * 100, color="#EF4444", lw=1.5, ls="--", label=f"ML均值 {res['return'].mean()*100:+.2f}%")
ax2.axvline(0, color="gray", lw=0.8)
ax2.set_title(f"每次換倉報酬分佈（{REBAL}日）")
ax2.set_xlabel("報酬率（%）")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

ax3 = axes[1, 0]
drawdown = (ml_cum + 1) / (ml_cum + 1).cummax() - 1
ax3.fill_between(drawdown.index, drawdown * 100, 0, alpha=0.4, color="#EF4444")
ax3.plot(drawdown.index, drawdown * 100, color="#DC2626", lw=1.2)
ax3.set_title(f"回撤曲線（最大回撤 {ml_maxdd*100:.1f}%）")
ax3.set_ylabel("回撤（%）")
ax3.grid(alpha=0.3)

ax4 = axes[1, 1]
alpha_cum = ml_cum - bm_cum.reindex(ml_cum.index).ffill()
ax4.plot(alpha_cum.index, alpha_cum * 100, color="#1D9E75", lw=1.5)
ax4.axhline(0, color="gray", lw=0.8, ls="--")
ax4.fill_between(alpha_cum.index, alpha_cum * 100, 0, where=alpha_cum >= 0, alpha=0.15, color="#1D9E75")
ax4.fill_between(alpha_cum.index, alpha_cum * 100, 0, where=alpha_cum < 0, alpha=0.15, color="#EF4444")
ax4.set_title("累積超額報酬（Alpha）")
ax4.set_ylabel("vs 基準（%）")
ax4.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "day19_ml_backtest.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n圖表已存檔：{out}")