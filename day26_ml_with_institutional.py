# day26_ml_with_institutional.py — 加入籌碼面的 ML 回測

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from FinMind.data import DataLoader
from datetime import date, timedelta
from watchlist import ALL_STOCKS
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

dl = DataLoader()
COST_RATE = 0.001425 + 0.003
TOP_N     = 5
REBAL     = 5

# ── 1. 技術面特徵函式 ─────────────────────────
def calc_rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l)

def build_tech_features(price, volume):
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

# ── 2. 籌碼面特徵函式 ─────────────────────────
def get_institutional(stock_id, days=400):
    start = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        df = dl.taiwan_stock_institutional_investors(
            stock_id=stock_id, start_date=start,
        )
        if df.empty:
            return None
        df["net"] = df["buy"] - df["sell"]
        pivot = df.pivot_table(
            index="date", columns="name", values="net", aggfunc="sum"
        )
        pivot.index = pd.to_datetime(pivot.index)
        col_map = {
            "Foreign_Investor": "foreign_net",
            "Investment_Trust": "trust_net",
            "Dealer_self":      "dealer_net",
        }
        pivot = pivot.rename(columns=col_map)
        for col in ["foreign_net","trust_net","dealer_net"]:
            if col not in pivot.columns:
                pivot[col] = 0
        result = pivot[["foreign_net","trust_net","dealer_net"]].fillna(0)

        # 標準化（除以20日平均成交量，讓不同市值股票可比較）
        for col in ["foreign_net","trust_net","dealer_net"]:
            rolling_std = result[col].rolling(20).std().replace(0, 1)
            result[f"{col}_z"] = result[col] / rolling_std

        result["total_net_z"] = (
            result["foreign_net_z"] +
            result["trust_net_z"] +
            result["dealer_net_z"]
        )
        result["foreign_5d_z"] = result["foreign_net_z"].rolling(5).mean()
        result["trust_5d_z"]   = result["trust_net_z"].rolling(5).mean()

        return result[[
            "foreign_net_z","trust_net_z","dealer_net_z",
            "total_net_z","foreign_5d_z","trust_5d_z"
        ]]
    except:
        return None

# ── 3. 下載資料 ───────────────────────────────
print(f"下載 {len(ALL_STOCKS)} 檔股價資料...")
raw     = yf.download(list(ALL_STOCKS.keys()), period="3y",
                      progress=False, auto_adjust=True)
prices  = raw["Close"].rename(columns=ALL_STOCKS).ffill().dropna(axis=1, thresh=200)
volumes = raw["Volume"].rename(columns=ALL_STOCKS).ffill()
prices, volumes = prices.align(volumes[prices.columns], join="inner")
names   = list(prices.columns)
print(f"可用股票：{len(names)} 檔")

# ── 4. 下載籌碼資料 ───────────────────────────
print("\n下載三大法人資料（約需 1～2 分鐘）...")
inst_data = {}
code_map  = {v: k for k, v in ALL_STOCKS.items()}

for i, name in enumerate(names):
    code = code_map.get(name, "").replace(".TW","").replace(".TWO","")
    if not code:
        continue
    data = get_institutional(code, days=400)
    if data is not None:
        inst_data[name] = data
    if (i+1) % 10 == 0:
        print(f"  已完成 {i+1}/{len(names)} 檔...")

print(f"成功取得籌碼資料：{len(inst_data)} 檔")

# ── 5. 建立特徵面板 ───────────────────────────
TECH_FEATURES  = ["r1","r5","r20","ma5_ratio","ma20_ratio","ma60_ratio",
                   "vol_ratio","vol_5d","rsi14","bb_pos","near_high"]
INST_FEATURES  = ["foreign_net_z","trust_net_z","dealer_net_z",
                   "total_net_z","foreign_5d_z","trust_5d_z"]
ALL_FEATURES   = TECH_FEATURES + INST_FEATURES

print("\n建立特徵資料集...")
frames = []
for name in names:
    tech = build_tech_features(prices[name], volumes[name])
    tech["stock"] = name

    if name in inst_data:
        inst = inst_data[name]
        tech = tech.join(inst, how="left")
    else:
        for col in INST_FEATURES:
            tech[col] = 0.0

    tech[INST_FEATURES] = tech[INST_FEATURES].fillna(0)
    frames.append(tech)

panel = pd.concat(frames).dropna(subset=TECH_FEATURES + ["future_5d"])
print(f"總樣本：{len(panel)} 筆")

# ── 6. 訓練 / 測試切割 ────────────────────────
dates      = sorted(panel.index.unique())
split_date = dates[int(len(dates) * 0.7)]
train = panel[panel.index <= split_date]
test  = panel[panel.index >  split_date]
print(f"訓練：{dates[0].date()} ～ {split_date.date()}")
print(f"測試：{split_date.date()} ～ {dates[-1].date()}")

# ── 7. 訓練兩個模型對比 ───────────────────────
print("\n訓練模型...")

# 純技術面
model_tech = RandomForestRegressor(
    n_estimators=300, max_depth=6,
    min_samples_leaf=20, random_state=42, n_jobs=-1
)
model_tech.fit(train[TECH_FEATURES], train["future_5d"])

# 技術面 + 籌碼面
model_full = RandomForestRegressor(
    n_estimators=300, max_depth=6,
    min_samples_leaf=20, random_state=42, n_jobs=-1
)
model_full.fit(train[ALL_FEATURES], train["future_5d"])

# ── 8. 回測函式 ───────────────────────────────
def run_backtest(model, features, label):
    test_copy = test.copy()
    test_copy["pred"] = model.predict(test_copy[features])
    rebal_dates = [d for i, d in enumerate(sorted(test_copy.index.unique()))
                   if i % REBAL == 0]
    results, prev = [], []
    for i, date_r in enumerate(rebal_dates[:-1]):
        next_date = rebal_dates[i + 1]
        day_data  = test_copy.loc[date_r].sort_values("pred", ascending=False)
        selected  = day_data["stock"].iloc[:TOP_N].tolist()
        rets = []
        for s in selected:
            try:
                rets.append((prices[s].loc[next_date] - prices[s].loc[date_r])
                            / prices[s].loc[date_r])
            except:
                rets.append(0.0)
        turnover = len(set(selected) - set(prev)) / TOP_N
        net_ret  = np.mean(rets) - turnover * COST_RATE
        results.append({"date": date_r, "return": net_ret,
                        "holdings": selected})
        prev = selected
    res = pd.DataFrame(results).set_index("date")
    cum = (1 + res["return"]).cumprod() - 1
    annual = (1 + res["return"].mean()) ** (252 / REBAL) - 1
    sharpe = res["return"].mean() / res["return"].std() * np.sqrt(252 / REBAL)
    maxdd  = ((cum+1)/(cum+1).cummax()-1).min()
    print(f"\n── {label} ──")
    print(f"年化報酬：{annual*100:+.1f}%  Sharpe：{sharpe:.2f}  最大回撤：{maxdd*100:.1f}%")
    return res, cum

res_tech, cum_tech = run_backtest(model_tech, TECH_FEATURES, "純技術面")
res_full, cum_full = run_backtest(model_full, ALL_FEATURES,  "技術+籌碼")

# ── 9. 特徵重要性 ─────────────────────────────
imp = pd.Series(model_full.feature_importances_, index=ALL_FEATURES)
imp = imp.sort_values(ascending=False)

# ── 10. 畫圖 ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("加入籌碼面後的模型績效比較", fontsize=13, fontweight="bold")

ax1 = axes[0]
ax1.plot(cum_tech.index, cum_tech*100,
         color="#94A3B8", lw=1.5, ls="--", label=f"純技術面 {cum_tech.iloc[-1]*100:+.1f}%")
ax1.plot(cum_full.index, cum_full*100,
         color="#7C3AED", lw=1.8, label=f"技術+籌碼 {cum_full.iloc[-1]*100:+.1f}%")
ax1.axhline(0, color="gray", lw=0.5)
ax1.set_title("累積報酬對比")
ax1.set_ylabel("累積報酬（%）")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

ax2 = axes[1]
colors_imp = ["#7C3AED" if i in INST_FEATURES else "#AFA9EC"
              for i in imp.index]
ax2.barh(imp.index[::-1], imp.values[::-1], color=colors_imp[::-1], alpha=0.85)
ax2.set_title("特徵重要性（深色=籌碼特徵）")
ax2.set_xlabel("重要性")
ax2.grid(axis="x", alpha=0.3)

plt.tight_layout()
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "day26_ml_institutional.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n圖表已存檔：{out}")