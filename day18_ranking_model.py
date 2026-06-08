"""Cross-sectional stock ranking model.

This script trains a RandomForestRegressor to rank Taiwan stocks by expected
5-day forward return. It evaluates the model with a time-based train/test split,
plots ranking diagnostics, and prints the latest model ranking.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor


warnings.filterwarnings("ignore")


DEFAULT_PERIOD = "3y"
DEFAULT_OUTPUT = "day18_ranking.png"
DEFAULT_CACHE_DIR = ".yfinance_cache"
FORWARD_DAYS = 5
TEST_SIZE = 0.2
TOP_N = 5
RANDOM_STATE = 42

STOCKS = {
    "2330.TW": "\u53f0\u7a4d\u96fb",
    "2317.TW": "\u9d3b\u6d77",
    "2454.TW": "\u806f\u767c\u79d1",
    "2308.TW": "\u53f0\u9054\u96fb",
    "2382.TW": "\u5ee3\u9054",
    "2303.TW": "\u806f\u96fb",
    "2412.TW": "\u4e2d\u83ef\u96fb",
    "2882.TW": "\u570b\u6cf0\u91d1",
    "1301.TW": "\u53f0\u5851",
    "2357.TW": "\u83ef\u78a9",
    "2886.TW": "\u5146\u8c50\u91d1",
    "3711.TW": "\u65e5\u6708\u5149\u6295\u63a7",
    "2002.TW": "\u4e2d\u92fc",
    "2881.TW": "\u5bcc\u90a6\u91d1",
    "2301.TW": "\u5149\u5bf6\u79d1",
}

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
    latest_ranking: pd.Series


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "Microsoft JhengHei"
    plt.rcParams["axes.unicode_minus"] = False


def configure_yfinance_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))


def download_market_data(symbols: list[str], period: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download close and volume data for all symbols."""
    raw = yf.download(symbols, period=period, progress=False, auto_adjust=False)

    if raw.empty:
        raise ValueError("No market data downloaded")

    prices = _extract_price_field(raw, "Close", symbols)
    volumes = _extract_price_field(raw, "Volume", symbols)
    prices = prices.rename(columns=STOCKS).dropna(how="all")
    volumes = volumes.rename(columns=STOCKS).dropna(how="all")

    if prices.empty or volumes.empty:
        raise ValueError("Downloaded data does not contain usable close and volume fields")

    return prices, volumes


def _extract_price_field(raw: pd.DataFrame, field: str, symbols: list[str]) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        data = raw[field].copy()
    else:
        data = raw[[field]].copy()
        data.columns = symbols[:1]

    return data.reindex(columns=symbols)


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
    features["bb_pos"] = (price - (bb_mid - 2 * bb_std)) / (4 * bb_std)
    features["near_high"] = price / price.rolling(20).max() - 1
    features["future_5d"] = price.pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)

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
    clean = dataset.dropna(subset=FEATURE_COLUMNS + ["future_5d"]).copy()
    dates = clean.index.get_level_values("Date").unique().sort_values()

    if len(dates) < 30:
        raise ValueError("Not enough dates to create a reliable train/test split")

    split_at = int(len(dates) * (1 - test_size))
    train_dates = dates[:split_at]
    test_dates = dates[split_at:]

    train = clean.loc[clean.index.get_level_values("Date").isin(train_dates)]
    test = clean.loc[clean.index.get_level_values("Date").isin(test_dates)]

    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty dataset")

    return train, test


def train_model(train_data: pd.DataFrame) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(train_data[FEATURE_COLUMNS], train_data["future_5d"])
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
        long_return = ranked["future_5d"].head(top_n).mean()
        short_return = ranked["future_5d"].tail(top_n).mean()
        long_short_return = long_return - short_return
        spear = safe_spearman(ranked["predicted"], ranked["future_5d"])

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


def predict_latest_ranking(model: RandomForestRegressor, panel: pd.DataFrame) -> tuple[pd.Timestamp, pd.Series]:
    """Predict the latest available cross-section and return scores by ticker."""
    latest_features = panel.dropna(subset=FEATURE_COLUMNS)
    latest_date = latest_features.index.get_level_values("Date").max()
    latest_cross_section = latest_features.xs(latest_date, level="Date").copy()
    latest_cross_section["predicted"] = model.predict(latest_cross_section[FEATURE_COLUMNS])
    latest_ranking = latest_cross_section["predicted"].sort_values(ascending=False)
    return latest_date, latest_ranking


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

    print("\nRanking model summary / \u6392\u540d\u6a21\u578b\u6458\u8981")
    print(f"Train rows / \u8a13\u7df4\u8cc7\u6599\u7b46\u6578: {len(result.train_data):,}")
    print(f"Test rows / \u6e2c\u8a66\u8cc7\u6599\u7b46\u6578:  {len(result.test_data):,}")
    print(f"Test days / \u6e2c\u8a66\u5929\u6578:  {len(daily):,}")
    print(f"Mean Spearman rank IC / \u5e73\u5747 Spearman \u6392\u540d IC: {daily['spearman'].mean():+.4f}")
    print(
        f"Long top {top_n} cumulative return / \u505a\u591a\u524d {top_n} \u540d\u7d2f\u7a4d\u5831\u916c: "
        f"{long_curve.iloc[-1] * 100:+.1f}%"
    )
    print(
        f"Short bottom {top_n} cumulative return / \u653e\u7a7a\u5f8c {top_n} \u540d\u7d2f\u7a4d\u5831\u916c: "
        f"{short_curve.iloc[-1] * 100:+.1f}%"
    )
    print(f"Long-short cumulative return / \u591a\u7a7a\u7d44\u5408\u7d2f\u7a4d\u5831\u916c: {long_short_curve.iloc[-1] * 100:+.1f}%")
    print(f"Mean long-short 5d return / \u5e73\u5747\u591a\u7a7a 5 \u65e5\u5831\u916c: {daily['long_short'].mean() * 100:+.4f}%")

    print("\nTop feature importances / \u91cd\u8981\u7279\u5fb5\u6392\u884c:")
    for name, value in result.feature_importances.head(8).items():
        print(f"  {name:<12} {value:.4f}")

    print("\n\u6700\u65b0\u6a21\u578b\u6392\u540d")
    print(f"\u65e5\u671f: {result.latest_date:%Y-%m-%d}")
    print(f"\n\u524d {top_n} \u540d / \u6a21\u578b\u9810\u671f\u5831\u916c\u8f03\u9ad8")
    for rank, (ticker, score) in enumerate(result.latest_ranking.head(top_n).items(), start=1):
        print(f"  {rank:2d}. {ticker:<16} \u5206\u6578={score:+.5f} [\u524d\u6bb5]")

    print(f"\n\u5f8c {top_n} \u540d / \u6a21\u578b\u9810\u671f\u5831\u916c\u8f03\u4f4e")
    bottom_ranking = result.latest_ranking.tail(top_n).sort_values(ascending=True)
    for rank, (ticker, score) in enumerate(bottom_ranking.items(), start=1):
        print(f"  {rank:2d}. {ticker:<16} \u5206\u6578={score:+.5f} [\u5f8c\u6bb5]")


def create_figure(result: RankingResult) -> plt.Figure:
    daily = result.daily_results
    long_curve = (1 + daily["long"]).cumprod() - 1
    short_curve = (1 + daily["short"]).cumprod() - 1
    long_short_curve = (1 + daily["long_short"]).cumprod() - 1

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("\u6a6b\u65b7\u9762\u6392\u540d\u6a21\u578b\u8a3a\u65b7", fontsize=13, fontweight="bold")

    ax1 = axes[0, 0]
    ax1.plot(long_short_curve.index, long_short_curve * 100, color="#7C3AED", lw=1.6, label="\u591a\u7a7a")
    ax1.plot(long_curve.index, long_curve * 100, color="#16A34A", lw=1.2, ls="--", label="\u505a\u591a")
    ax1.plot(short_curve.index, short_curve * 100, color="#DC2626", lw=1.2, ls="--", label="\u653e\u7a7a")
    ax1.axhline(0, color="gray", lw=0.8, ls=":")
    ax1.set_title("\u6e2c\u8a66\u671f\u7d2f\u7a4d\u5831\u916c")
    ax1.set_ylabel("\u5831\u916c\u7387 (%)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2 = axes[0, 1]
    rolling_spearman = daily["spearman"].rolling(20).mean()
    ax2.plot(daily.index, daily["spearman"], color="#AFA9EC", lw=0.8, alpha=0.45, label="\u6bcf\u65e5 Spearman")
    ax2.plot(rolling_spearman.index, rolling_spearman, color="#534AB7", lw=1.6, label="20\u65e5\u5e73\u5747")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.axhline(0.05, color="#16A34A", lw=1, ls=":", label="+0.05")
    ax2.axhline(-0.05, color="#DC2626", lw=1, ls=":", label="-0.05")
    ax2.set_title(f"\u6392\u540d IC\uff0c\u5e73\u5747={daily['spearman'].mean():+.4f}")
    ax2.set_ylabel("Spearman \u6392\u540d\u76f8\u95dc")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ax3 = axes[1, 0]
    importances = result.feature_importances.sort_values()
    colors = ["#7C3AED" if value > importances.median() else "#AFA9EC" for value in importances]
    ax3.barh(importances.index, importances.values, color=colors, alpha=0.85)
    ax3.set_title("\u7279\u5fb5\u91cd\u8981\u6027")
    ax3.set_xlabel("\u91cd\u8981\u6027")
    ax3.grid(axis="x", alpha=0.3)

    ax4 = axes[1, 1]
    ax4.hist(daily["long_short"] * 100, bins=40, color="#7C3AED", alpha=0.7, label="\u591a\u7a7a")
    ax4.axvline(daily["long_short"].mean() * 100, color="#EF4444", lw=1.5, ls="--", label="\u5e73\u5747")
    ax4.axvline(0, color="gray", lw=0.8)
    ax4.set_title("\u591a\u7a7a 5 \u65e5\u5831\u916c\u5206\u5e03")
    ax4.set_xlabel("\u5831\u916c\u7387 (%)")
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
    print(f"Downloading / \u6b63\u5728\u4e0b\u8f09 {len(symbols)} symbols / \u6a94\u80a1\u7968, period / \u671f\u9593={args.period}...")
    result = run_pipeline(symbols, args.period, args.test_size, args.top_n)
    print_summary(result, args.top_n)

    fig = create_figure(result)
    output_path = args.output.resolve()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()

    print(f"\nChart saved / \u5716\u8868\u5df2\u5132\u5b58: {output_path}")


if __name__ == "__main__":
    main()
