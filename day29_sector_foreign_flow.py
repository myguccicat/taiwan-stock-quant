# day29_sector_foreign_flow.py — 加入產業群組外資流向特徵
# 在 day28_sector_ranking.py 基礎上新增：
#   sector_foreign_flow : 該股所屬產業群組的外資近5日累積買超（標準化）
#   foreign_vs_sector   : 個股外資買超相對同群組的強弱

from __future__ import annotations
import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from FinMind.data import DataLoader
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_PERIOD  = "3y"
DEFAULT_OUTPUT  = "day29_sector_foreign_flow.png"
FORWARD_DAYS    = 3
TEST_SIZE       = 0.2
TOP_N           = 5
RANDOM_STATE    = 42
INST_DAYS       = 400   # 抓幾天的三大法人資料

from watchlist import ALL_STOCKS
STOCKS = ALL_STOCKS

# 產業分群（與 day28 相同）
SECTOR_MAP = {
    "聯發科": "A_chip", "世芯-KY": "A_chip", "訊芯-KY": "A_chip",
    "晶心科": "A_chip", "智原": "A_chip", "M31": "A_chip",
    "聯詠": "A_chip", "原相": "A_chip", "光寶科": "A_chip", "所羅門": "A_chip",
    "南亞科": "B_memory", "華邦電": "B_memory", "旺宏": "B_memory",
    "台積電": "C_osat", "日月光投控": "C_osat", "力成": "C_osat",
    "穩懋": "C_osat", "超豐": "C_osat", "IET-KY": "C_osat",
    "台表科": "C_osat", "全新": "C_osat",
    "南電": "D_pcb", "欣興": "D_pcb", "臻鼎-KY": "D_pcb",
    "家登": "D_pcb", "弘塑": "D_pcb",
    "廣達": "E_server", "英業達": "E_server", "緯創": "E_server",
    "鴻海": "E_server", "緯穎": "E_server", "樺漢": "E_server", "研華": "E_server",
    "奇鋐": "F_thermal", "貿聯-KY": "F_thermal",
    "國巨": "G_passive", "凱美": "G_passive", "尼克森": "G_passive",
    "志聖": "H_equip", "中砂": "H_equip", "家碩": "H_equip",
    "商丞": "H_equip", "鈦昇": "H_equip", "意德士": "H_equip",
    "技嘉": "I_other", "華碩": "I_other", "群創": "I_other",
    "宏達電": "I_other", "晟銘電": "I_other", "倉佑": "I_other",
    "東陽": "I_other", "文曄": "I_other", "慧洋-KY": "I_other",
}

TECH_FEATURES = [
    "r1", "r5", "r20", "ma5_ratio", "ma20_ratio", "ma60_ratio",
    "vol_ratio", "vol_5d", "rsi14", "bb_pos", "near_high",
]
SECTOR_FEATURES = ["sector_momentum", "sector_rank", "inter_sector_rank"]
INST_FEATURES   = ["sector_foreign_flow", "foreign_vs_sector"]
ALL_FEATURES    = TECH_FEATURES + SECTOR_FEATURES + INST_FEATURES


@dataclass(frozen=True)
class RankingResult:
    model: RandomForestRegressor
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    daily_results: pd.DataFrame
    feature_importances: pd.Series
    latest_date: pd.Timestamp
    latest_ranking: pd.DataFrame


def download_market_data(symbols, period):
    raw = yf.download(symbols, period=period, progress=False,
                      auto_adjust=True, group_by="ticker")
    prices_dict, volumes_dict = {}, {}
    for sym in symbols:
        name = STOCKS[sym]
        try:
            if sym in raw.columns.get_level_values(0):
                prices_dict[name]  = raw[sym]["Close"]
                volumes_dict[name] = raw[sym]["Volume"]
        except Exception:
            pass
    prices  = pd.DataFrame(prices_dict).ffill().bfill()
    volumes = pd.DataFrame(volumes_dict).ffill().bfill()
    return prices, volumes


def calc_rsi(prices, window=14):
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))


def build_tech_features(price, volume):
    f = pd.DataFrame(index=price.index)
    r1 = price.pct_change(1)
    f["r1"]         = r1
    f["r5"]         = price.pct_change(5)
    f["r20"]        = price.pct_change(20)
    f["ma5_ratio"]  = price / price.rolling(5).mean()  - 1
    f["ma20_ratio"] = price / price.rolling(20).mean() - 1
    f["ma60_ratio"] = price / price.rolling(60).mean() - 1
    f["vol_ratio"]  = volume / volume.rolling(20).mean()
    f["vol_5d"]     = r1.rolling(5).std()
    f["rsi14"]      = calc_rsi(price)
    bb_mid = price.rolling(20).mean()
    bb_std = price.rolling(20).std()
    f["bb_pos"]     = (price - (bb_mid - 2*bb_std)) / (4*bb_std).replace(0, np.nan)
    f["near_high"]  = price / price.rolling(20).max() - 1
    f["future_3d"]  = price.pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)
    return f.replace([np.inf, -np.inf], np.nan)


def fetch_institutional_data(names: list[str]) -> dict:
    """抓取多檔股票的外資買賣超，回傳 {name: pd.Series(net_foreign, index=date)}"""
    dl = DataLoader()
    start = (date.today() - timedelta(days=INST_DAYS)).strftime("%Y-%m-%d")
    code_map = {v: k.replace(".TW","").replace(".TWO","") for k, v in STOCKS.items()}

    inst_dict = {}
    success, failed = 0, 0

    print(f"  抓取外資資料（{len(names)} 檔，約需 2-3 分鐘）...")
    for name in names:
        code = code_map.get(name)
        if not code:
            continue
        try:
            df = dl.taiwan_stock_institutional_investors(
                stock_id=code, start_date=start
            )
            if df.empty:
                failed += 1
                continue
            # 只取外資買賣超
            foreign = df[df["name"] == "Foreign_Investor"].copy()
            foreign["net"] = foreign["buy"] - foreign["sell"]
            series = foreign.groupby("date")["net"].sum()
            series.index = pd.to_datetime(series.index)
            inst_dict[name] = series
            success += 1
        except Exception:
            failed += 1

    print(f"  外資資料：成功 {success} 檔，失敗 {failed} 檔")
    return inst_dict


def add_all_features(panel: pd.DataFrame, inst_dict: dict) -> pd.DataFrame:
    """新增產業特徵 + 外資流向特徵"""
    panel = panel.copy()
    panel["sector"] = panel.index.get_level_values("Ticker").map(SECTOR_MAP).fillna("Z_unknown")

    # ── 先加外資買賣超到 panel ──────────────────────
    panel["foreign_net"] = np.nan
    for name, series in inst_dict.items():
        mask = panel.index.get_level_values("Ticker") == name
        # 對齊日期，用 ffill 填補缺漏的交易日
        aligned = series.reindex(panel[mask].index.get_level_values("Date"), method="ffill")
        panel.loc[mask, "foreign_net"] = aligned.values

    panel["foreign_net"] = panel["foreign_net"].fillna(0)

    # ── 按日期計算所有特徵 ──────────────────────────
    results = []
    for dt, day_df in panel.groupby(level="Date"):
        day_df = day_df.copy()

        # 1. sector_rank
        day_df["sector_rank"] = day_df.groupby("sector")["r1"].rank(pct=True)

        # 2. sector_momentum
        day_df["sector_momentum"] = day_df.groupby("sector")["r5"].transform("mean")

        # 3. inter_sector_rank
        sector_means   = day_df.groupby("sector")["r5"].mean()
        stock_sec_mean = day_df["sector"].map(sector_means)
        day_df["inter_sector_rank"] = (day_df["r5"] - stock_sec_mean).rank(pct=True)

        # 4. sector_foreign_flow：群組外資近5日標準化買超
        #    （這裡用當日的 foreign_net 加總代替，真實應用需 rolling，簡化處理）
        sec_foreign = day_df.groupby("sector")["foreign_net"].transform("sum")
        sec_foreign_std = sec_foreign.std()
        day_df["sector_foreign_flow"] = (
            sec_foreign / sec_foreign_std if sec_foreign_std > 0 else 0
        )

        # 5. foreign_vs_sector：個股外資 - 群組均值
        sec_foreign_mean = day_df.groupby("sector")["foreign_net"].transform("mean")
        diff = day_df["foreign_net"] - sec_foreign_mean
        diff_std = diff.std()
        day_df["foreign_vs_sector"] = diff / diff_std if diff_std > 0 else 0

        results.append(day_df)

    return pd.concat(results).drop(columns=["sector", "foreign_net"])


def build_panel(prices, volumes, inst_dict):
    frames = []
    for ticker in prices.columns:
        f = build_tech_features(prices[ticker], volumes[ticker])
        f["Ticker"] = ticker
        frames.append(f)
    panel = pd.concat(frames)
    panel.index.name = "Date"
    panel = panel.set_index("Ticker", append=True).sort_index()

    print("  計算產業 + 外資流向特徵...")
    panel = add_all_features(panel, inst_dict)
    return panel


def split_train_test(panel):
    clean = panel.dropna(subset=ALL_FEATURES + ["future_3d"]).copy()
    dates = clean.index.get_level_values("Date").unique().sort_values()
    split = int(len(dates) * (1 - TEST_SIZE))
    train_d = dates[:split]
    test_d  = dates[split:]
    train = clean.loc[clean.index.get_level_values("Date").isin(train_d)]
    test  = clean.loc[clean.index.get_level_values("Date").isin(test_d)]
    return train, test


def train_model(train):
    model = RandomForestRegressor(
        n_estimators=300, max_depth=6,
        min_samples_leaf=20, random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(train[ALL_FEATURES], train["future_3d"])
    return model


def evaluate(test, preds):
    scored = test.copy()
    scored["predicted"] = preds
    rows = []
    for dt, group in scored.groupby(level="Date"):
        if len(group) < TOP_N * 2:
            continue
        ranked = group.sort_values("predicted", ascending=False)
        rows.append({
            "date":       dt,
            "long":       ranked["future_3d"].head(TOP_N).mean(),
            "short":      ranked["future_3d"].tail(TOP_N).mean(),
            "long_short": ranked["future_3d"].head(TOP_N).mean() - ranked["future_3d"].tail(TOP_N).mean(),
            "spearman":   spearmanr(ranked["predicted"], ranked["future_3d"])[0]
                          if ranked["predicted"].nunique() > 1 else np.nan,
        })
    return pd.DataFrame(rows).set_index("date").dropna()


def predict_latest(model, panel):
    lf = panel.dropna(subset=ALL_FEATURES)
    latest_date = lf.index.get_level_values("Date").max()
    cross = lf.xs(latest_date, level="Date").copy()
    cross["predicted"] = model.predict(cross[ALL_FEATURES])
    cross["Rank"] = cross["predicted"].rank(ascending=False, method="first").astype(int)
    return latest_date, cross.sort_values("Rank").reset_index()


def print_summary(result):
    daily = result.daily_results
    lc = (1 + daily["long"]).cumprod() - 1
    sc = (1 + daily["short"]).cumprod() - 1
    ls = (1 + daily["long_short"]).cumprod() - 1

    print("\n" + "="*62)
    print("📈 橫斷面排名模型（技術面 + 產業特徵 + 外資流向）")
    print("="*62)
    print(f"訓練集：{len(result.train_data):,}  |  測試天數：{len(daily)}")
    print(f"Spearman Rank IC：{daily['spearman'].mean():+.4f}")
    print(f"做多前{TOP_N}名累積報酬：{lc.iloc[-1]*100:+.1f}%")
    print(f"放空後{TOP_N}名累積報酬：{sc.iloc[-1]*100:+.1f}%")
    print(f"多空組合累積報酬：{ls.iloc[-1]*100:+.1f}%")

    print(f"\n特徵重要性（前10）：")
    for name, val in result.feature_importances.head(10).items():
        tag = ""
        if name in SECTOR_FEATURES: tag = " ★產業"
        if name in INST_FEATURES:   tag = " ★外資"
        print(f"  {name:<25} {val*100:.2f}%{tag}")

    ranking_df = result.latest_ranking
    today_str  = result.latest_date.strftime("%Y-%m-%d")
    n_total    = len(ranking_df)

    print(f"\n{'🔥'*5} 決策面板 ({today_str}) {'🔥'*5}")
    print(f"\n👑 前 {TOP_N} 名：")
    for _, row in ranking_df.head(TOP_N).iterrows():
        sec = SECTOR_MAP.get(row["Ticker"], "其他")
        print(f"  第{int(row['Rank']):2d}名 | {row['Ticker']:<15} | {row['predicted']:+.5f} | {sec}")

    print(f"\n🛡️ 緩衝區（第 {TOP_N+1}~10 名）：")
    for _, row in ranking_df.iloc[TOP_N:10].iterrows():
        sec = SECTOR_MAP.get(row["Ticker"], "其他")
        print(f"  第{int(row['Rank']):2d}名 | {row['Ticker']:<15} | {row['predicted']:+.5f} | {sec}")

    print(f"\n🛑 警報區（後 {TOP_N} 名）：")
    for _, row in ranking_df.tail(TOP_N).iloc[::-1].iterrows():
        rev = n_total - int(row["Rank"]) + 1
        sec = SECTOR_MAP.get(row["Ticker"], "其他")
        print(f"  倒數第{rev:2d}名 | {row['Ticker']:<15} | {row['predicted']:+.5f} | {sec}")
    print("="*62)

    out_dir  = Path(__file__).with_name("stock")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"predictions_foreign_{today_str}.csv"
    ranking_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"💾 存檔：{csv_path}")


def create_figure(result, output_path):
    daily = result.daily_results
    lc = (1 + daily["long"]).cumprod() - 1
    sc = (1 + daily["short"]).cumprod() - 1
    ls = (1 + daily["long_short"]).cumprod() - 1

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("橫斷面排名模型（技術面 + 產業 + 外資流向）診斷圖",
                 fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(ls.index, ls*100, color="#7C3AED", lw=1.6, label="多空組合")
    ax.plot(lc.index, lc*100, color="#16A34A", lw=1.2, ls="--", label=f"前{TOP_N}名做多")
    ax.plot(sc.index, sc*100, color="#DC2626", lw=1.2, ls="--", label=f"後{TOP_N}名放空")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("測試期累積報酬")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    rolling_ic = daily["spearman"].rolling(20).mean()
    ax.plot(daily.index, daily["spearman"], color="#AFA9EC", lw=0.8, alpha=0.45)
    ax.plot(rolling_ic.index, rolling_ic, color="#534AB7", lw=1.6, label="20日均")
    ax.axhline(0,    color="gray", lw=0.8, ls="--")
    ax.axhline(0.05, color="#16A34A", lw=1, ls=":")
    ax.axhline(-0.05, color="#DC2626", lw=1, ls=":")
    ax.set_title(f"Rank IC（平均={daily['spearman'].mean():+.4f}）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    imp = result.feature_importances.sort_values()
    colors = []
    for n in imp.index:
        if n in INST_FEATURES:   colors.append("#EF4444")   # 紅=外資
        elif n in SECTOR_FEATURES: colors.append("#7C3AED") # 紫=產業
        else:                      colors.append("#AFA9EC")  # 灰=技術
    ax.barh(imp.index, imp.values*100, color=colors, alpha=0.85)
    ax.set_title("特徵重要性（紅=外資，紫=產業，灰=技術）")
    ax.set_xlabel("重要性（%）")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1, 1]
    ax.hist(daily["long_short"]*100, bins=40, color="#7C3AED", alpha=0.7)
    ax.axvline(daily["long_short"].mean()*100, color="#EF4444", lw=1.5, ls="--", label="平均")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_title("多空組合 3日報酬分布")
    ax.set_xlabel("報酬率（%）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"圖表已存檔：{output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name(DEFAULT_OUTPUT))
    args = parser.parse_args()

    yf.set_tz_cache_location(str(Path(__file__).with_name(".yfinance_cache")))

    symbols = list(STOCKS.keys())
    names   = list(STOCKS.values())

    print(f"下載 {len(symbols)} 檔股價資料...")
    prices, volumes = download_market_data(symbols, args.period)
    print(f"可用股票：{len(prices.columns)} 檔，{len(prices)} 個交易日")

    # 抓外資資料（只抓有在 SECTOR_MAP 的股票）
    sector_names = [n for n in names if n in SECTOR_MAP]
    inst_dict = fetch_institutional_data(sector_names)

    print("建立特徵資料集...")
    panel = build_panel(prices, volumes, inst_dict)

    train, test = split_train_test(panel)
    print(f"訓練：{train.index.get_level_values('Date').min().date()} ～ "
          f"{train.index.get_level_values('Date').max().date()}")
    print(f"測試：{test.index.get_level_values('Date').min().date()} ～ "
          f"{test.index.get_level_values('Date').max().date()}")

    print("訓練模型...")
    model  = train_model(train)
    preds  = model.predict(test[ALL_FEATURES])
    daily  = evaluate(test, preds)
    imp    = pd.Series(model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
    latest_date, ranking = predict_latest(model, panel)

    result = RankingResult(
        model=model, train_data=train, test_data=test,
        daily_results=daily, feature_importances=imp,
        latest_date=latest_date, latest_ranking=ranking,
    )

    print_summary(result)
    create_figure(result, args.output)


if __name__ == "__main__":
    main()
