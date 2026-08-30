# day32_window5_diagnostic.py — 深挖 walk-forward 視窗5的選股差異
#
# 背景：day31 顯示 ML 模型在視窗5（訓練至2026-02-06，測試2026-02-09~2026-05-18）
# 大幅甩開簡單版 sector_momentum 規則（超額 +83.8% vs -13.7%），但其他5個視窗
# 兩者表現相近，甚至簡單版更好。這支腳本把視窗5每天實際選出的股票攤開來看，
# 分成「只有ML選、naive沒選」「只有naive選、ML沒選」「兩邊都選」三組，
# 各自統計出現次數跟平均未來3日報酬，藉此看出：
#   (a) ML的優勢是不是由少數幾檔「naive完全沒抓到」的股票撐起來的
#   (b) 這些股票分布在哪些產業——集中在一兩個產業，還是分散在很多產業
#       （分散代表比較像真正的個股選擇能力，集中則可能只是換了一個
#        naive規則沒抓到的強勢產業，本質上還是產業動能）
#
# 執行：python day32_window5_diagnostic.py
# 可選參數：--window-index（預設5，對齊 day30/day31 印出的「視窗 N/6」）
#          --min-train-days / --test-days / --step-days（要跟 day30/day31
#          用同一組參數，才能對到同一個視窗；預設值已跟兩者的預設值一致）

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from day28_sector_ranking import (
    STOCKS,
    SECTOR_MAP,
    ALL_FEATURES,
    TOP_N,
    DEFAULT_CACHE_PATH,
    configure_yfinance_cache,
    load_or_download_market_data,
    build_panel,
    train_model,
)
from day30_walkforward import generate_walkforward_windows
from day31_naive_vs_model import naive_predictions

DEFAULT_PERIOD         = "3y"
DEFAULT_MIN_TRAIN_DAYS = 300
DEFAULT_TEST_DAYS      = 60
DEFAULT_STEP_DAYS      = 60
DEFAULT_WINDOW_INDEX   = 5


def get_window(
    panel: pd.DataFrame, window_index: int, min_train_days: int, test_days: int, step_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = panel.dropna(subset=ALL_FEATURES + ["future_3d"]).copy()
    dates = clean.index.get_level_values("Date").unique().sort_values()
    windows = generate_walkforward_windows(dates, min_train_days, test_days, step_days)
    if not windows:
        raise ValueError("資料量不足，產生不出任何 walk-forward 視窗")
    if window_index < 1 or window_index > len(windows):
        raise ValueError(f"只有 {len(windows)} 個視窗（1~{len(windows)}），無法取得第 {window_index} 個")
    train_dates, test_dates = windows[window_index - 1]
    train = clean.loc[clean.index.get_level_values("Date").isin(train_dates)]
    test  = clean.loc[clean.index.get_level_values("Date").isin(test_dates)]
    return train, test


def daily_top_picks(scored: pd.DataFrame, top_n: int) -> dict[pd.Timestamp, list[str]]:
    """回傳 {日期: [前N名股票代碼]}，只保留候選數足夠的交易日（跟 evaluate() 篩選邏輯一致）"""
    picks: dict[pd.Timestamp, list[str]] = {}
    for date, group in scored.groupby(level="Date"):
        if len(group) < top_n * 2:
            continue
        top = group.sort_values("predicted", ascending=False).head(top_n)
        picks[date] = list(top.index.get_level_values("Ticker"))
    return picks


def summarize_exclusive_picks(
    test: pd.DataFrame, ml_picks: dict, naive_picks: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int]]:
    """把每天的選股分成三組：只有ML選、只有naive選、兩邊都選，
    各自記錄股票、日期、當天的未來3日報酬。同時記錄每天的重疊檔數，供觀察選股分歧程度。"""
    ml_only_rows, naive_only_rows, both_rows = [], [], []
    overlap_counts = []

    common_dates = sorted(set(ml_picks) & set(naive_picks))
    for date in common_dates:
        ml_set    = set(ml_picks[date])
        naive_set = set(naive_picks[date])
        overlap_counts.append(len(ml_set & naive_set))

        day_data = test.xs(date, level="Date")

        for ticker in ml_set - naive_set:
            ret = day_data.loc[ticker, "future_3d"] if ticker in day_data.index else np.nan
            ml_only_rows.append({"date": date, "ticker": ticker, "future_3d": ret})
        for ticker in naive_set - ml_set:
            ret = day_data.loc[ticker, "future_3d"] if ticker in day_data.index else np.nan
            naive_only_rows.append({"date": date, "ticker": ticker, "future_3d": ret})
        for ticker in ml_set & naive_set:
            ret = day_data.loc[ticker, "future_3d"] if ticker in day_data.index else np.nan
            both_rows.append({"date": date, "ticker": ticker, "future_3d": ret})

    return (
        pd.DataFrame(ml_only_rows),
        pd.DataFrame(naive_only_rows),
        pd.DataFrame(both_rows),
        overlap_counts,
    )


def aggregate_ticker_stats(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame(columns=["ticker", "count", "avg_future_3d", "sector"])
    agg = rows_df.groupby("ticker")["future_3d"].agg(["count", "mean"]).reset_index()
    agg = agg.rename(columns={"mean": "avg_future_3d"})
    agg["sector"] = agg["ticker"].map(SECTOR_MAP).fillna("其他")
    return agg.sort_values(["count", "avg_future_3d"], ascending=[False, False])


def print_group_summary(name: str, rows_df: pd.DataFrame, top_k: int = 10) -> None:
    print(f"\n【{name}】")
    if rows_df.empty:
        print("　（沒有任何選股落在這組）")
        return
    total_occurrences = len(rows_df)
    avg_return = rows_df["future_3d"].mean()
    print(f"　總選股次數：{total_occurrences}　平均未來3日報酬：{avg_return*100:+.2f}%")

    agg = aggregate_ticker_stats(rows_df)
    print(f"　出現最多次的股票（前{min(top_k, len(agg))}檔）：")
    for _, row in agg.head(top_k).iterrows():
        print(
            f"　  {row['ticker']:<15} 出現{int(row['count']):2d}次　"
            f"平均未來3日報酬 {row['avg_future_3d']*100:+.2f}%　產業：{row['sector']}"
        )

    sector_counts = agg.groupby("sector")["count"].sum().sort_values(ascending=False)
    n_sectors = len(sector_counts)
    top_sector_share = sector_counts.iloc[0] / sector_counts.sum() * 100 if n_sectors else 0
    print(f"　涉及產業數：{n_sectors}　最集中的產業佔比：{top_sector_share:.0f}%（{sector_counts.index[0] if n_sectors else '—'}）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--window-index", type=int, default=DEFAULT_WINDOW_INDEX,
                        help="要診斷第幾個 walk-forward 視窗（1-based，對齊 day30/day31 的印出編號）")
    parser.add_argument("--min-train-days", type=int, default=DEFAULT_MIN_TRAIN_DAYS)
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument("--step-days", type=int, default=DEFAULT_STEP_DAYS)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH,
                        help="資料快取檔位置，跟 day28/30/31/33 共用同一份才能互相比較")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="強制重新下載資料並覆蓋快取（預設是有快取就直接用）")
    args = parser.parse_args()

    configure_yfinance_cache(Path(__file__).with_name(".yfinance_cache"))

    symbols = list(STOCKS.keys())
    print(f"下載 {len(symbols)} 檔資料...")
    prices, volumes = load_or_download_market_data(
        symbols, args.period, args.cache_path, args.refresh_cache
    )
    print(f"可用股票：{len(prices.columns)} 檔，{len(prices)} 個交易日")

    print("建立特徵資料集（含產業特徵）...")
    panel = build_panel(prices, volumes)

    train, test = get_window(
        panel, args.window_index, args.min_train_days, args.test_days, args.step_days
    )
    test_start = test.index.get_level_values("Date").min().date()
    test_end   = test.index.get_level_values("Date").max().date()
    print(f"\n鎖定視窗 {args.window_index}：測試期 {test_start} ～ {test_end}（{test.index.get_level_values('Date').nunique()} 個交易日）")

    print("\n訓練 ML 模型...")
    model = train_model(train)
    preds_ml = model.predict(test[ALL_FEATURES])
    scored_ml = test.copy()
    scored_ml["predicted"] = preds_ml

    preds_naive = naive_predictions(test)
    scored_naive = test.copy()
    scored_naive["predicted"] = preds_naive

    ml_picks    = daily_top_picks(scored_ml, TOP_N)
    naive_picks = daily_top_picks(scored_naive, TOP_N)

    ml_only, naive_only, both, overlap_counts = summarize_exclusive_picks(test, ml_picks, naive_picks)

    print("\n" + "=" * 60)
    print(f"📊 視窗 {args.window_index} 選股差異診斷（前{TOP_N}名，ML模型 vs 簡單版sector_momentum規則）")
    print("=" * 60)

    if overlap_counts:
        avg_overlap = np.mean(overlap_counts)
        print(f"平均每天重疊檔數：{avg_overlap:.1f} / {TOP_N} 檔"
              f"（重疊越少，代表兩邊選股邏輯差異越大）")

    print_group_summary(f"只有 ML 選（naive完全沒選到）", ml_only)
    print_group_summary(f"只有簡單版選（ML完全沒選到）", naive_only)
    print_group_summary(f"兩邊都選", both)

    if not ml_only.empty and not naive_only.empty:
        edge = ml_only["future_3d"].mean() - naive_only["future_3d"].mean()
        print(f"\n【關鍵對比】ML獨有選股 vs naive獨有選股，平均未來3日報酬差距：{edge*100:+.2f}%")
        if edge > 0:
            print("　→ ML挑出的、naive完全沒抓到的股票，報酬明顯更好，是這個視窗ML勝出的主要來源")
        else:
            print("　→ ML獨有的選股報酬反而不比naive獨有的好，優勢可能來自別的地方"
                  "（例如ML更常挑到『兩邊都選』裡報酬較好的那些，或是選股權重/排序上的細微差異）")

    print(f"\n這個視窗訓練出來的模型，特徵重要性排行：")
    imp = pd.Series(model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
    for name, val in imp.head(8).items():
        print(f"  {name:<20} {val*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
