"""Cross-sectional stock ranking model (Optimized for 15W Real Trading).

This script trains a RandomForestRegressor to rank Taiwan stocks by expected
3-day forward return. It evaluates the model, implements a buffer rank check,
and auto-archives the daily predictions into a desktop folder to prevent column alignment bugs.
"""

from __future__ import annotations

import argparse
import warnings
import os
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor


warnings.filterwarnings("ignore")


DEFAULT_PERIOD = "3y"
DEFAULT_OUTPUT = "day27_ranking_3day.png"  # 對齊最新 3 日換倉圖表
DEFAULT_CACHE_DIR = ".yfinance_cache"
FORWARD_DAYS = 3  # 🔥 實戰優化：改為 3 日換倉預測目標
TEST_SIZE = 0.2
TOP_N = 5
RANDOM_STATE = 42

from watchlist import ALL_STOCKS
STOCKS = ALL_STOCKS

FEATURE_COLUMNS = [
    "r1",
    "r5",
    "r20",
    "ma5_ratio",
    "ma20_ratio",
    "ma60_ratio",
    "vol_ratio",
    "vol_5d",
    "rsi14",
    "bb_pos",
    "near_high",
]


@dataclass(frozen=True)
class RankingResult:
    model: RandomForestRegressor
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    daily_results: pd.DataFrame
    feature_importances: pd.Series
    latest_date: pd.Timestamp
    latest_ranking: pd.DataFrame  # 升級：回傳包含完整特徵與代碼對齊的 DataFrame


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "Microsoft JhengHei"
    plt.rcParams["axes.unicode_minus"] = False


def configure_yfinance_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))


def download_market_data(symbols: list[str], period: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download close and volume data for all symbols."""
    # group_by="ticker" 能徹底阻絕多個商品下載時 MultiIndex 欄位交叉錯位的 Bug
    raw = yf.download(symbols, period=period, progress=False, auto_adjust=True, group_by="ticker")

    if raw.empty:
        raise ValueError("No market data downloaded")

    prices_dict = {}
    volumes_dict = {}

    for sym in symbols:
        try:
            if sym in raw.columns.levels[0]:
                prices_dict[STOCKS[sym]] = raw[sym]["Close"]
                volumes_dict[STOCKS[sym]] = raw[sym]["Volume"]
        except Exception:
            # 兼容單一股票或非 MultiIndex 狀況
            if len(symbols) == 1:
                prices_dict[STOCKS[sym]] = raw["Close"]
                volumes_dict[STOCKS[sym]] = raw["Volume"]

    prices = pd.DataFrame(prices_dict).ffill().bfill()
    volumes = pd.DataFrame(volumes_dict).ffill().bfill()

    if prices.empty or volumes.empty:
        raise ValueError("Downloaded data does not contain usable close and volume fields")

    return prices, volumes


def calc_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def build_symbol_features(price: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """Build technical features for one symbol."""
    features = pd.DataFrame(index=price.index)
    returns_1d = price.pct_change(1)

    features["r1"] = returns_1d
    features["r5"] = price.pct_change(5)
    features["r20"] = price.pct_change(20)
    features["ma5_ratio"] = price / price.rolling(5).mean() - 1
    features["ma20_ratio"] = price / price.rolling(20).mean() - 1
    features["ma60_ratio"] = price / price.rolling(60).mean() - 1
    features["vol_ratio"] = volume / volume.rolling(20).mean()
    features["vol_5d"] = returns_1d.rolling(5).std()
    features["rsi14"] = calc_rsi(price)

    bb_mid = price.rolling(20).mean()
    bb_std = price.rolling(20).std()
    features["bb_pos"] = (price - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)
    features["near_high"] = price / price.rolling(20).max() - 1
    
    # 🔥 實戰優化：預測目標錨定為未來 3 日報酬率
    features["future_3d"] = price.pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)

    return features.replace([np.inf, -np.inf], np.nan)


def build_panel_dataset(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """Build a MultiIndex DataFrame indexed by Date and Ticker."""
    symbol_frames = []

    for ticker in prices.columns:
        feature_frame = build_symbol_features(prices[ticker], volumes[ticker])
        feature_frame["Ticker"] = ticker
        symbol_frames.append(feature_frame)

    panel = pd.concat(symbol_frames)
    panel.index.name = "Date"
    panel = panel.set_index("Ticker", append=True).sort_index()
    return panel


def split_train_test(dataset: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date so every test row is later than every train row."""
    clean = dataset.dropna(subset=FEATURE_COLUMNS + ["future_3d"]).copy()
    dates = clean.index.get_level_values("Date").unique().sort_values()

    if len(dates) < 30:
        raise ValueError("Not enough dates to create a reliable train/test split")

    split_at = int(len(dates) * (1 - test_size))
    train_dates = dates[:split_at]
    test_dates = dates[split_at:]

    train = clean.loc[clean.index.get_level_values("Date").isin(train_dates)]
    test = clean.loc[clean.index.get_level_values("Date").isin(test_dates)]

    return train, test


def train_model(train_data: pd.DataFrame) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(train_data[FEATURE_COLUMNS], train_data["future_3d"])
    return model


def evaluate_ranking(test_data: pd.DataFrame, predictions: np.ndarray, top_n: int) -> pd.DataFrame:
    """Evaluate daily long-short ranking quality on the test set."""
    scored = test_data.copy()
    scored["predicted"] = predictions
    rows = []

    for date, group in scored.groupby(level="Date"):
        if len(group) < top_n * 2:
            continue

        ranked = group.sort_values("predicted", ascending=False)
        long_return = ranked["future_3d"].head(top_n).mean()
        short_return = ranked["future_3d"].tail(top_n).mean()
        long_short_return = long_return - short_return
        spear = safe_spearman(ranked["predicted"], ranked["future_3d"])

        rows.append(
            {
                "date": date,
                "long": long_return,
                "short": short_return,
                "long_short": long_short_return,
                "spearman": spear,
            }
        )

    results = pd.DataFrame(rows).set_index("date")
    return results.dropna()


def safe_spearman(predicted: pd.Series, actual: pd.Series) -> float:
    if predicted.nunique() < 2 or actual.nunique() < 2:
        return np.nan

    value, _ = spearmanr(predicted, actual)
    return float(value)


def predict_latest_ranking(model: RandomForestRegressor, panel: pd.DataFrame) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Predict the latest available cross-section and return rich DataFrame with rank."""
    latest_features = panel.dropna(subset=FEATURE_COLUMNS)
    latest_date = latest_features.index.get_level_values("Date").max()
    latest_cross_section = latest_features.xs(latest_date, level="Date").copy()
    
    # 預測並打分
    latest_cross_section["predicted"] = model.predict(latest_cross_section[FEATURE_COLUMNS])
    latest_cross_section["Rank"] = latest_cross_section["predicted"].rank(ascending=False, method="first").astype(int)
    
    final_ranking = latest_cross_section.sort_values(by="Rank").reset_index()
    return latest_date, final_ranking


def run_pipeline(symbols: list[str], period: str, test_size: float, top_n: int) -> RankingResult:
    prices, volumes = download_market_data(symbols, period)
    panel = build_panel_dataset(prices, volumes)
    train_data, test_data = split_train_test(panel, test_size)
    model = train_model(train_data)
    predictions = model.predict(test_data[FEATURE_COLUMNS])
    daily_results = evaluate_ranking(test_data, predictions, top_n)

    if daily_results.empty:
        raise ValueError("Ranking evaluation produced no usable daily results")

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    latest_date, latest_ranking = predict_latest_ranking(model, panel)

    return RankingResult(
        model=model,
        train_data=train_data,
        test_data=test_data,
        daily_results=daily_results,
        feature_importances=importances,
        latest_date=latest_date,
        latest_ranking=latest_ranking,
    )


def print_summary(result: RankingResult, top_n: int) -> None:
    daily = result.daily_results
    long_curve = (1 + daily["long"]).cumprod() - 1
    short_curve = (1 + daily["short"]).cumprod() - 1
    long_short_curve = (1 + daily["long_short"]).cumprod() - 1

    print("\n" + "="*60)
    print("📈 ML 橫斷面選股模型實戰摘要 (3日目標優化版)")
    print("="*60)
    print(f"訓練集樣本數: {len(result.train_data):,}")
    print(f"測試集樣本數: {len(result.test_data):,}")
    print(f"測試總交易日: {len(daily):,}")
    print(f"平均資訊值 (Mean Spearman Rank IC): {daily['spearman'].mean():+.4f}")
    print(f"做多前 {top_n} 名累積報酬: {long_curve.iloc[-1] * 100:+.1f}%")
    print(f"放空後 {top_n} 名累積報酬: {short_curve.iloc[-1] * 100:+.1f}%")
    print(f"多空組合累積報酬:       {long_short_curve.iloc[-1] * 100:+.1f}%")

    print("\n💡 核心特徵重要性排行 (Feature Importances):")
    for name, value in result.feature_importances.head(5).items():
        print(f"  {name:<12} {value * 100:.2f}%")

    # 🔥 實戰控制面板輸出
    ranking_df = result.latest_ranking
    today_str = result.latest_date.strftime("%Y-%m-%d")
    
    print("\n" + "🔥" * 5 + f" 15萬實戰跟單決策面板 ({today_str}) " + "🔥" * 5)
    print(f"\n👑 建議買進/續抱核心【前 {top_n} 名】(等權重 20% 分配):")
    for _, row in ranking_df.head(top_n).iterrows():
        print(f"  第 {int(row['Rank']):2d} 名 | Ticker: {row['Ticker']:<10} | 預測得分: {row['predicted']:+.5f}")

    print(f"\n🛡️ 智慧緩衝留守區【第 6 ~ 10 名】(原持有則不更換):")
    for _, row in ranking_df.iloc[top_n:10].iterrows():
        print(f"  第 {int(row['Rank']):2d} 名 | Ticker: {row['Ticker']:<10} | 預測得分: {row['predicted']:+.5f}")

    print(f"\n🛑 警報避開區【最後 {top_n} 名】(切勿買進):")
    for _, row in ranking_df.tail(top_n).iloc[::-1].iterrows():
        print(f"  倒數第 {int(len(ranking_df) - row['Rank'] + 1):2d} 名 | Ticker: {row['Ticker']:<10} | 預測得分: {row['predicted']:+.5f}")
    print("="*60)

    # 💾 實戰優化：自動備份為 CSV 供下輪換倉時比對 Rank 緩衝區
    desktop_stock_dir = Path(__file__).with_name("stock")
    desktop_stock_dir.mkdir(parents=True, exist_ok=True)
    csv_path = desktop_stock_dir / f"predictions_{today_str}.csv"
    ranking_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"💾 決策歷史紀錄已自動封存至：{csv_path}\n")


def create_figure(result: RankingResult) -> plt.Figure:
    daily = result.daily_results
    long_curve = (1 + daily["long"]).cumprod() - 1
    short_curve = (1 + daily["short"]).cumprod() - 1
    long_short_curve = (1 + daily["long_short"]).cumprod() - 1

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("橫斷面 3日排名模型實戰診斷圖", fontsize=14, fontweight="bold")

    ax1 = axes[0, 0]
    ax1.plot(long_short_curve.index, long_short_curve * 100, color="#7C3AED", lw=1.6, label="多空組合")
    ax1.plot(long_curve.index, long_curve * 100, color="#16A34A", lw=1.2, ls="--", label="最優前5名做多")
    ax1.plot(short_curve.index, short_curve * 100, color="#DC2626", lw=1.2, ls="--", label="最劣後5名放空")
    ax1.axhline(0, color="gray", lw=0.8, ls=":")
    ax1.set_title("測試期累積報酬走勢")
    ax1.set_ylabel("報酬率 (%)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2 = axes[0, 1]
    rolling_spearman = daily["spearman"].rolling(20).mean()
    ax2.plot(daily.index, daily["spearman"], color="#AFA9EC", lw=0.8, alpha=0.45, label="每日 Spearman IC")
    ax2.plot(rolling_spearman.index, rolling_spearman, color="#534AB7", lw=1.6, label="20日移動平均")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.axhline(0.05, color="#16A34A", lw=1, ls=":", label="+0.05 顯著正相關")
    ax2.axhline(-0.05, color="#DC2626", lw=1, ls=":", label="-0.05 顯著負相關")
    ax2.set_title(f"預報能力分析 (Rank IC 平均={daily['spearman'].mean():+.4f})")
    ax2.set_ylabel("Spearman 相關係數")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ax3 = axes[1, 0]
    importances = result.feature_importances.sort_values()
    colors = ["#7C3AED" if value > importances.median() else "#AFA9EC" for value in importances]
    ax3.barh(importances.index, importances.values * 100, color=colors, alpha=0.85)
    ax3.set_title("機器人選股核心审美 (特徵重要性)")
    ax3.set_xlabel("重要性權重 (%)")
    ax3.grid(axis="x", alpha=0.3)

    ax4 = axes[1, 1]
    ax4.hist(daily["long_short"] * 100, bins=40, color="#7C3AED", alpha=0.7, label="多空組合")
    ax4.axvline(daily["long_short"].mean() * 100, color="#EF4444", lw=1.5, ls="--", label="平均值")
    ax4.axvline(0, color="gray", lw=0.8)
    ax4.set_title("多空組合 3日 報酬率隨機分佈")
    ax4.set_xlabel("報酬率 (%)")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a stock ranking model")
    parser.add_argument("--period", default=DEFAULT_PERIOD, help=f"Yahoo Finance download period, default: {DEFAULT_PERIOD}")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name(DEFAULT_OUTPUT), help="Output PNG path")
    parser.add_argument("--top-n", type=int, default=TOP_N, help=f"Number of long and short names per test day, default: {TOP_N}")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help=f"Fraction of dates used for testing, default: {TEST_SIZE}")
    parser.add_argument("--no-show", action="store_true", help="Save the chart without opening a GUI window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    configure_yfinance_cache(Path(__file__).with_name(DEFAULT_CACHE_DIR))

    symbols = list(STOCKS.keys())
    result = run_pipeline(symbols, args.period, args.test_size, args.top_n)
    print_summary(result, args.top_n)

    fig = create_figure(result)
    output_path = args.output.resolve()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()