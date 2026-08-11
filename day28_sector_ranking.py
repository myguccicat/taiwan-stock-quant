# day28_sector_ranking.py — 加入產業分類特徵的排名模型
# 在 day27_ranking_3day.py 基礎上新增三個產業輪動特徵：
#   sector_rank      : 個股在自己群組內的相對排名（0~1，越高越強）
#   sector_momentum  : 群組整體近5日平均報酬
#   inter_sector_rank: 個股相對所有群組平均的強弱

from __future__ import annotations
import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

# ── 設定 ──────────────────────────────────────────
DEFAULT_PERIOD   = "3y"
DEFAULT_OUTPUT   = "day28_sector_ranking.png"
FORWARD_DAYS     = 3
TEST_SIZE        = 0.2
TOP_N            = 5
RANDOM_STATE     = 42
# ──────────────────────────────────────────────────

from watchlist import ALL_STOCKS
STOCKS = ALL_STOCKS

# 產業分群（股票名稱 → 群組代碼）
SECTOR_MAP = {
    # 晶片設計
    "聯發科": "晶片設計", "世芯-KY": "晶片設計", "訊芯-KY": "晶片設計",
    "晶心科": "晶片設計", "智原": "晶片設計", "M31": "晶片設計",
    "聯詠": "晶片設計", "原相": "晶片設計", "光寶科": "晶片設計", "所羅門": "晶片設計",
    "凌陽科技股份有限公司": "晶片設計", "威盛電子股份有限公司": "晶片設計",
    "新唐科技股份有限公司": "晶片設計", "晶豪科技股份有限公司": "晶片設計",
    "盛群半導體股份有限公司": "晶片設計", "矽統科技股份有限公司": "晶片設計",
    "迅杰科技股份有限公司": "晶片設計",

    # 記憶體
    "南亞科": "記憶體", "華邦電": "記憶體", "旺宏": "記憶體",
    "力晶積成電子製造股份有限公司": "記憶體",
    "宇瞻科技股份有限公司": "記憶體", "創見資訊股份有限公司": "記憶體",

    # 製造封測
    "台積電": "製造封測", "日月光投控": "製造封測", "力成": "製造封測",
    "穩懋": "製造封測", "超豐": "製造封測", "IET-KY": "製造封測",
    "台表科": "製造封測", "全新": "製造封測",
    "京元電子股份有限公司": "製造封測", "南茂科技股份有限公司": "製造封測",
    "矽格股份有限公司": "製造封測", "菱生精密工業股份有限公司": "製造封測",
    "超豐電子股份有限公司": "製造封測", "嘉晶電子股份有限公司": "製造封測",

    # PCB載板
    "南電": "PCB載板", "欣興": "PCB載板", "臻鼎-KY": "PCB載板",
    "家登": "PCB載板", "弘塑": "PCB載板",
    "景碩科技股份有限公司": "PCB載板", "台塑勝高科技股份有限公司": "PCB載板",
    "福懋科技股份有限公司": "PCB載板", "同欣電子工業股份有限公司": "PCB載板",
    "華東科技股份有限公司": "PCB載板",

    # AI伺服器
    "廣達": "AI伺服器", "英業達": "AI伺服器", "緯創": "AI伺服器",
    "鴻海": "AI伺服器", "緯穎": "AI伺服器", "樺漢": "AI伺服器", "研華": "AI伺服器",
    "仁寶電腦工業股份有限公司": "AI伺服器", "和碩聯合科技股份有限公司": "AI伺服器",
    "佳世達科技股份有限公司": "AI伺服器", "神達控股股份有限公司": "AI伺服器",
    "永擎": "AI伺服器",

    # 散熱電源
    "奇鋐": "散熱電源", "貿聯-KY": "散熱電源",

    # 被動元件
    "國巨": "被動元件", "凱美": "被動元件", "尼克森": "被動元件",
    "富鼎先進電子股份有限公司": "被動元件", "強茂股份有限公司": "被動元件",
    "承啟科技股份有限公司": "被動元件",

    # 半導體設備
    "志聖": "半導體設備", "中砂": "半導體設備", "家碩": "半導體設備",
    "商丞": "半導體設備", "鈦昇": "半導體設備", "意德士": "半導體設備",
    "事欣科技股份有限公司": "半導體設備",

    # 其他電子
    "技嘉": "其他電子", "華碩": "其他電子", "群創": "其他電子",
    "宏達電": "其他電子", "晟銘電": "其他電子", "倉佑": "其他電子",
    "東陽": "其他電子", "文曄": "其他電子", "慧洋-KY": "其他電子",
    "全友電腦股份有限公司": "其他電子", "宏碁股份有限公司": "其他電子",
    "微星科技股份有限公司": "其他電子", "聯華電子股份有限公司": "其他電子",
    "國巨*": "其他電子", "群益半導體收益": "其他電子",
    "台塑石化股份有限公司": "其他電子", "台灣汽電共生股份有限公司": "其他電子",

    # 光電
    "聯鈞光電股份有限公司": "光電",
}

TECH_FEATURES = [
    "r1", "r5", "r20", "ma5_ratio", "ma20_ratio", "ma60_ratio",
    "vol_ratio", "vol_5d", "rsi14", "bb_pos", "near_high",
]

SECTOR_FEATURES = [
    "sector_rank",
    "sector_momentum",
    "inter_sector_rank",
]

ALL_FEATURES = TECH_FEATURES + SECTOR_FEATURES


@dataclass(frozen=True)
class RankingResult:
    model: RandomForestRegressor
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    daily_results: pd.DataFrame
    feature_importances: pd.Series
    latest_date: pd.Timestamp
    latest_ranking: pd.DataFrame


def configure_yfinance_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))


def download_market_data(symbols: list[str], period: str):
    raw = yf.download(symbols, period=period, progress=False,
                      auto_adjust=True, group_by="ticker")
    if raw.empty:
        raise ValueError("No market data downloaded")

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


def calc_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))


def build_tech_features(price: pd.Series, volume: pd.Series) -> pd.DataFrame:
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


def add_sector_features(panel: pd.DataFrame) -> pd.DataFrame:
    """新增三個產業輪動特徵"""
    panel = panel.copy()

    # 取得每檔股票的群組
    panel["sector"] = panel.index.get_level_values("Ticker").map(SECTOR_MAP).fillna("Z_unknown")

    # 按日期分組計算
    results = []
    for date, day_df in panel.groupby(level="Date"):
        day_df = day_df.copy()

        # 1. sector_rank：個股在自己群組內的 r1 排名（0~1）
        day_df["sector_rank"] = day_df.groupby("sector")["r1"].rank(pct=True)

        # 2. sector_momentum：群組整體近5日平均 r5
        sector_avg_r5 = day_df.groupby("sector")["r5"].transform("mean")
        day_df["sector_momentum"] = sector_avg_r5

        # 3. inter_sector_rank：個股 r5 相對所有群組均值的排名
        sector_means = day_df.groupby("sector")["r5"].mean()
        # 每股所屬群組的均值
        stock_sector_mean = day_df["sector"].map(sector_means)
        # 個股 r5 - 所屬群組均值 = 超出群組的部分
        day_df["inter_sector_rank"] = (
            day_df["r5"] - stock_sector_mean
        ).rank(pct=True)

        results.append(day_df)

    return pd.concat(results).drop(columns=["sector"])


def build_panel(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ticker in prices.columns:
        f = build_tech_features(prices[ticker], volumes[ticker])
        f["Ticker"] = ticker
        frames.append(f)
    panel = pd.concat(frames)
    panel.index.name = "Date"
    panel = panel.set_index("Ticker", append=True).sort_index()

    # 加入產業特徵
    print("  計算產業輪動特徵...")
    panel = add_sector_features(panel)
    return panel


def split_train_test(panel: pd.DataFrame):
    clean  = panel.dropna(subset=ALL_FEATURES + ["future_3d"]).copy()
    dates  = clean.index.get_level_values("Date").unique().sort_values()
    split  = int(len(dates) * (1 - TEST_SIZE))
    train_dates = dates[:split]
    test_dates  = dates[split:]
    train = clean.loc[clean.index.get_level_values("Date").isin(train_dates)]
    test  = clean.loc[clean.index.get_level_values("Date").isin(test_dates)]
    return train, test


def train_model(train: pd.DataFrame) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=300, max_depth=6,
        min_samples_leaf=20, random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(train[ALL_FEATURES], train["future_3d"])
    return model


def evaluate(test: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    scored = test.copy()
    scored["predicted"] = preds
    rows = []
    for date, group in scored.groupby(level="Date"):
        if len(group) < TOP_N * 2:
            continue
        ranked = group.sort_values("predicted", ascending=False)
        rows.append({
            "date":       date,
            "long":       ranked["future_3d"].head(TOP_N).mean(),
            "short":      ranked["future_3d"].tail(TOP_N).mean(),
            "long_short": ranked["future_3d"].head(TOP_N).mean() - ranked["future_3d"].tail(TOP_N).mean(),
            "spearman":   spearmanr(ranked["predicted"], ranked["future_3d"])[0]
                          if ranked["predicted"].nunique() > 1 else np.nan,
        })
    return pd.DataFrame(rows).set_index("date").dropna()


def predict_latest(model: RandomForestRegressor, panel: pd.DataFrame):
    latest_features = panel.dropna(subset=ALL_FEATURES)
    latest_date = latest_features.index.get_level_values("Date").max()
    cross = latest_features.xs(latest_date, level="Date").copy()
    cross["predicted"] = model.predict(cross[ALL_FEATURES])
    cross["Rank"] = cross["predicted"].rank(ascending=False, method="first").astype(int)
    ranking = cross.sort_values("Rank").reset_index()
    return latest_date, ranking


def print_summary(result: RankingResult) -> None:
    daily = result.daily_results
    lc = (1 + daily["long"]).cumprod() - 1
    sc = (1 + daily["short"]).cumprod() - 1
    ls = (1 + daily["long_short"]).cumprod() - 1

    print("\n" + "="*60)
    print("📈 橫斷面排名模型（含產業特徵）實戰摘要")
    print("="*60)
    print(f"訓練集：{len(result.train_data):,} 筆")
    print(f"測試集：{len(result.test_data):,} 筆  |  測試天數：{len(daily)}")
    print(f"Spearman Rank IC：{daily['spearman'].mean():+.4f}")
    print(f"做多前{TOP_N}名累積報酬：{lc.iloc[-1]*100:+.1f}%")
    print(f"放空後{TOP_N}名累積報酬：{sc.iloc[-1]*100:+.1f}%")
    print(f"多空組合累積報酬：{ls.iloc[-1]*100:+.1f}%")

    print(f"\n特徵重要性排行：")
    for name, val in result.feature_importances.head(8).items():
        tag = " ★產業特徵" if name in SECTOR_FEATURES else ""
        print(f"  {name:<20} {val*100:.2f}%{tag}")

    ranking_df  = result.latest_ranking
    today_str   = result.latest_date.strftime("%Y-%m-%d")
    n_total     = len(ranking_df)

    print(f"\n{'🔥'*5} 決策面板 ({today_str}) {'🔥'*5}")
    print(f"\n👑 前 {TOP_N} 名：")
    for _, row in ranking_df.head(TOP_N).iterrows():
        sector = SECTOR_MAP.get(row["Ticker"], "其他")
        print(f"  第{int(row['Rank']):2d}名 | {row['Ticker']:<15} | {row['predicted']:+.5f} | {sector}")

    print(f"\n🛡️ 緩衝區（第 {TOP_N+1}~10 名）：")
    for _, row in ranking_df.iloc[TOP_N:10].iterrows():
        sector = SECTOR_MAP.get(row["Ticker"], "其他")
        print(f"  第{int(row['Rank']):2d}名 | {row['Ticker']:<15} | {row['predicted']:+.5f} | {sector}")

    print(f"\n🛑 警報區（後 {TOP_N} 名）：")
    for _, row in ranking_df.tail(TOP_N).iloc[::-1].iterrows():
        rev_rank = n_total - int(row["Rank"]) + 1
        sector = SECTOR_MAP.get(row["Ticker"], "其他")
        print(f"  倒數第{rev_rank:2d}名 | {row['Ticker']:<15} | {row['predicted']:+.5f} | {sector}")
    print("="*60)

    # 存 CSV
    out_dir = Path(__file__).with_name("stock")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"predictions_sector_{today_str}.csv"
    ranking_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"💾 存檔：{csv_path}")

    # 自動記錄 IC 到追蹤表
    from datetime import date as _date
    ic_log_path = Path(__file__).with_name("ic_tracking.csv")
    ic_record = pd.DataFrame([{
        "date":    _date.today().strftime("%Y-%m-%d"),
        "ic_mean": round(daily["spearman"].mean(), 4),
        "ic_20d":  round(daily["spearman"].tail(20).mean(), 4),
        "stocks":  len(result.latest_ranking),
    }])
    if ic_log_path.exists():
        ic_record.to_csv(ic_log_path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        ic_record.to_csv(ic_log_path, index=False, encoding="utf-8-sig")
    print(f"📊 IC 追蹤記錄已更新：{ic_log_path}")


def create_figure(result: RankingResult, output_path: Path) -> None:
    daily = result.daily_results
    lc = (1 + daily["long"]).cumprod() - 1
    sc = (1 + daily["short"]).cumprod() - 1
    ls = (1 + daily["long_short"]).cumprod() - 1

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("橫斷面排名模型（含產業特徵）實戰診斷圖", fontsize=14, fontweight="bold")

    # 左上：累積報酬
    ax = axes[0, 0]
    ax.plot(ls.index, ls*100, color="#7C3AED", lw=1.6, label="多空組合")
    ax.plot(lc.index, lc*100, color="#16A34A", lw=1.2, ls="--", label=f"前{TOP_N}名做多")
    ax.plot(sc.index, sc*100, color="#DC2626", lw=1.2, ls="--", label=f"後{TOP_N}名放空")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("測試期累積報酬")
    ax.set_ylabel("報酬率（%）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 右上：Spearman IC
    ax = axes[0, 1]
    rolling_ic = daily["spearman"].rolling(20).mean()
    ax.plot(daily.index, daily["spearman"], color="#AFA9EC", lw=0.8, alpha=0.45)
    ax.plot(rolling_ic.index, rolling_ic, color="#534AB7", lw=1.6, label="20日均")
    ax.axhline(0,    color="gray", lw=0.8, ls="--")
    ax.axhline(0.05, color="#16A34A", lw=1, ls=":", label="+0.05")
    ax.axhline(-0.05, color="#DC2626", lw=1, ls=":", label="-0.05")
    ax.set_title(f"Rank IC（平均={daily['spearman'].mean():+.4f}）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 左下：特徵重要性
    ax = axes[1, 0]
    imp = result.feature_importances.sort_values()
    colors = ["#7C3AED" if n in SECTOR_FEATURES else "#AFA9EC" for n in imp.index]
    ax.barh(imp.index, imp.values*100, color=colors, alpha=0.85)
    ax.set_title("特徵重要性（深色=產業特徵）")
    ax.set_xlabel("重要性（%）")
    ax.grid(axis="x", alpha=0.3)

    # 右下：報酬分布
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


def run_pipeline(period: str):
    symbols = list(STOCKS.keys())

    print(f"下載 {len(symbols)} 檔資料...")
    prices, volumes = download_market_data(symbols, period)
    print(f"可用股票：{len(prices.columns)} 檔，{len(prices)} 個交易日")

    print("建立特徵資料集（含產業特徵）...")
    panel = build_panel(prices, volumes)

    train, test = split_train_test(panel)
    print(f"訓練：{train.index.get_level_values('Date').min().date()} ～ "
          f"{train.index.get_level_values('Date').max().date()}")
    print(f"測試：{test.index.get_level_values('Date').min().date()} ～ "
          f"{test.index.get_level_values('Date').max().date()}")

    print("訓練模型...")
    model = train_model(train)

    preds  = model.predict(test[ALL_FEATURES])
    daily  = evaluate(test, preds)
    imp    = pd.Series(model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
    latest_date, ranking = predict_latest(model, panel)

    return RankingResult(
        model=model, train_data=train, test_data=test,
        daily_results=daily, feature_importances=imp,
        latest_date=latest_date, latest_ranking=ranking,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name(DEFAULT_OUTPUT))
    args = parser.parse_args()

    configure_yfinance_cache(Path(__file__).with_name(".yfinance_cache"))

    result = run_pipeline(args.period)
    print_summary(result)
    create_figure(result, args.output)


if __name__ == "__main__":
    main()
