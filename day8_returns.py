"""Analyze daily return distribution for a Taiwan stock.

The script downloads price data from Yahoo Finance, computes daily returns,
prints summary statistics, and saves a four-panel chart to day8_returns.png.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats


DEFAULT_SYMBOL = "2330.TW"
DEFAULT_PERIOD = "2y"
DEFAULT_OUTPUT = "day8_returns.png"
DEFAULT_CACHE_DIR = ".yfinance_cache"
TRADING_DAYS_PER_MONTH = 21


@dataclass(frozen=True)
class ReturnSummary:
    mean_pct: float
    std_pct: float
    skew: float
    kurtosis: float
    threshold_pct: float
    big_up_count: int
    big_down_count: int


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "Microsoft JhengHei"
    plt.rcParams["axes.unicode_minus"] = False


def configure_yfinance_cache(cache_dir: Path) -> None:
    """Keep yfinance cache files in a writable project-local directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))


def download_close_prices(symbol: str, period: str) -> pd.DataFrame:
    """Download close prices and return a normalized DataFrame."""
    data = yf.download(symbol, period=period, progress=False, auto_adjust=False)

    if data.empty:
        raise ValueError(f"No price data found for {symbol} during {period}")

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    result = close.to_frame(name="close").dropna()
    if len(result) < 2:
        raise ValueError(f"Not enough close-price rows for {symbol} to compute returns")

    return result


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Add daily return data to close prices."""
    returns = prices.copy()
    returns["return"] = returns["close"].pct_change()
    returns = returns.dropna(subset=["return"])

    if returns.empty:
        raise ValueError("Daily return data is empty")

    return returns


def find_big_moves(returns: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Find trading days whose absolute return is above the threshold."""
    big_moves = returns[returns["return"].abs() > threshold].copy()
    big_moves["direction"] = np.where(big_moves["return"] > 0, "up", "down")
    return big_moves


def summarize_returns(returns: pd.DataFrame, big_moves: pd.DataFrame, threshold: float) -> ReturnSummary:
    daily_returns = returns["return"]

    return ReturnSummary(
        mean_pct=daily_returns.mean() * 100,
        std_pct=daily_returns.std() * 100,
        skew=daily_returns.skew(),
        kurtosis=daily_returns.kurt(),
        threshold_pct=threshold * 100,
        big_up_count=int((big_moves["direction"] == "up").sum()),
        big_down_count=int((big_moves["direction"] == "down").sum()),
    )


def print_report(symbol: str, returns: pd.DataFrame, big_moves: pd.DataFrame, summary: ReturnSummary) -> None:
    daily_returns = returns["return"]
    skew_note = "left-skewed; large downside moves dominate" if summary.skew < 0 else "right-skewed; large upside moves dominate"

    print(f"\n{symbol} daily return summary")
    print(f"Mean daily return: {summary.mean_pct:+.4f}%")
    print(f"Std. deviation:    {summary.std_pct:.4f}%")
    print(f"Skewness:          {summary.skew:.4f} ({skew_note})")
    print(f"Kurtosis:          {summary.kurtosis:.4f} (above 0 usually means fatter tails)")

    print(f"\nLarge-move threshold: +/-{summary.threshold_pct:.2f}%")
    print(f"Large-move days:      {len(big_moves)}")
    print(f"  Up days:            {summary.big_up_count}")
    print(f"  Down days:          {summary.big_down_count}")

    print("\nTop 5 daily gains:")
    print(daily_returns.nlargest(5).map(lambda value: f"{value * 100:+.2f}%").to_string())
    print("\nTop 5 daily losses:")
    print(daily_returns.nsmallest(5).map(lambda value: f"{value * 100:+.2f}%").to_string())


def plot_return_timeseries(
    ax: plt.Axes,
    returns: pd.DataFrame,
    big_moves: pd.DataFrame,
    threshold: float,
) -> None:
    daily_returns_pct = returns["return"] * 100
    big_up = big_moves[big_moves["direction"] == "up"]
    big_down = big_moves[big_moves["direction"] == "down"]

    ax.plot(returns.index, daily_returns_pct, color="#64748B", linewidth=0.7, alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(threshold * 100, color="#EF4444", linewidth=1, linestyle=":", label=f"+{threshold * 100:.1f}%")
    ax.axhline(-threshold * 100, color="#EF4444", linewidth=1, linestyle=":", label=f"-{threshold * 100:.1f}%")
    ax.scatter(big_up.index, big_up["return"] * 100, color="#16A34A", s=25, zorder=5)
    ax.scatter(big_down.index, big_down["return"] * 100, color="#DC2626", s=25, zorder=5)
    ax.set_title("Daily returns and large moves")
    ax.set_ylabel("Return (%)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_return_distribution(ax: plt.Axes, daily_returns: pd.Series) -> None:
    daily_returns_pct = daily_returns * 100
    x = np.linspace(daily_returns_pct.min(), daily_returns_pct.max(), 300)
    normal_curve = stats.norm.pdf(x, daily_returns_pct.mean(), daily_returns_pct.std())

    ax.hist(daily_returns_pct, bins=60, color="#2563EB", alpha=0.65, density=True, label="Actual")
    ax.plot(x, normal_curve, color="#EF4444", linewidth=1.5, linestyle="--", label="Normal")
    ax.set_title("Daily return distribution vs normal")
    ax.set_xlabel("Return (%)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_qq(ax: plt.Axes, daily_returns: pd.Series) -> None:
    (osm, osr), (slope, intercept, _) = stats.probplot(daily_returns, dist="norm")

    ax.scatter(osm, osr, color="#2563EB", s=8, alpha=0.5)
    ax.plot(osm, slope * np.asarray(osm) + intercept, color="#EF4444", linewidth=1.5, linestyle="--")
    ax.set_title("Q-Q plot")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.grid(alpha=0.3)


def plot_monthly_returns(ax: plt.Axes, returns: pd.DataFrame) -> None:
    monthly_returns = returns.groupby(returns.index.month)["return"].mean() * 100 * TRADING_DAYS_PER_MONTH
    colors = ["#16A34A" if value >= 0 else "#DC2626" for value in monthly_returns]

    ax.bar(monthly_returns.index, monthly_returns.values, color=colors, alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Average return by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly average return (%)")
    ax.set_xticks(range(1, 13))
    ax.grid(axis="y", alpha=0.3)


def create_figure(symbol: str, returns: pd.DataFrame, big_moves: pd.DataFrame, threshold: float) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"{symbol} daily return analysis", fontsize=13, fontweight="bold")

    plot_return_timeseries(axes[0, 0], returns, big_moves, threshold)
    plot_return_distribution(axes[0, 1], returns["return"])
    plot_qq(axes[1, 0], returns["return"])
    plot_monthly_returns(axes[1, 1], returns)

    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze daily returns and save a chart")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help=f"Yahoo Finance symbol, default: {DEFAULT_SYMBOL}")
    parser.add_argument("--period", default=DEFAULT_PERIOD, help=f"Download period, default: {DEFAULT_PERIOD}")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name(DEFAULT_OUTPUT), help="Output PNG path")
    parser.add_argument("--no-show", action="store_true", help="Save the chart without opening a GUI window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    configure_yfinance_cache(Path(__file__).with_name(DEFAULT_CACHE_DIR))

    print(f"Downloading price data for {args.symbol}...")
    prices = download_close_prices(args.symbol, args.period)
    returns = calculate_returns(prices)
    threshold = returns["return"].std() * 2
    big_moves = find_big_moves(returns, threshold)
    summary = summarize_returns(returns, big_moves, threshold)

    print_report(args.symbol, returns, big_moves, summary)

    fig = create_figure(args.symbol, returns, big_moves, threshold)
    output_path = args.output.resolve()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()

    print(f"\nChart saved: {output_path}")


if __name__ == "__main__":
    main()
