# day33_offset_sensitivity.py — 複利抽樣路徑敏感度分析
#
# 背景：day32 顯示視窗2「攤開所有天數看」ML獨有選股其實贏過naive獨有選股
# （+1.20% vs +0.82%），但day30/day31用「不重疊複利」算出來的視窗2超額報酬
# 卻是 -22.3%（輸很慘）。這兩個數字沒有互相矛盾，是評估方法不同：
# 不重疊複利只抽測試期裡每隔 FORWARD_DAYS（3）天的一個樣本點去做真正的
# 複利滾動（避免day28修過的重疊區間灌水bug），但「從哪一天開始抽」這件事
# 本身是任意的——只是剛好對齊測試期第一天。如果換一個起始點（offset），
# 抽到的樣本點、複利路徑都會完全不同，如果最終報酬因此劇烈跳動，代表單一
# 視窗的複利報酬數字本身就有相當大的雜訊成分，不完全代表「這段時間選股
# 品質」的真實差異——這件事不影響 day30 整體 83% 勝率／68% 捕捉比例的
# 可信度（那是跨6個視窗、更大樣本算出來的），但會影響單一視窗數字的解讀。
#
# 做法：對指定視窗，把 FORWARD_DAYS（3）種可能的起始offset都跑一次不重疊
# 複利，分別算出 ML 跟簡單版規則各自的超額報酬，比較這3個數字彼此差多少。
#
# 執行：python day33_offset_sensitivity.py --window-index 2
#      python day33_offset_sensitivity.py --window-index 5

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from day28_sector_ranking import (
    STOCKS,
    ALL_FEATURES,
    TOP_N,
    FORWARD_DAYS,
    DEFAULT_CACHE_PATH,
    configure_yfinance_cache,
    load_or_download_market_data,
    build_panel,
    train_model,
    evaluate,
)
from day31_naive_vs_model import naive_predictions
from day32_window5_diagnostic import get_window

DEFAULT_PERIOD         = "3y"
DEFAULT_MIN_TRAIN_DAYS = 300
DEFAULT_TEST_DAYS      = 60
DEFAULT_STEP_DAYS      = 60


def nonoverlap_mask_offset(daily: pd.DataFrame, offset: int) -> pd.Series:
    """跟 day28 的 nonoverlap_mask 邏輯一樣（每隔 FORWARD_DAYS 天取一個樣本，
    確保複利用的區間彼此不重疊），但起始點可以往後挪 offset 天。
    offset=0 就是 day28/day30/day31 原本用的版本（從測試期第一天開始抽）。"""
    idx = np.arange(len(daily))
    return pd.Series((idx >= offset) & ((idx - offset) % FORWARD_DAYS == 0), index=daily.index)


def compounded_excess_for_offset(
    test: pd.DataFrame, daily: pd.DataFrame, offset: int
) -> tuple[float, int] | None:
    """回傳 (超額報酬, 樣本數)，如果這個offset篩選後沒有可用樣本則回傳 None。"""
    mask = nonoverlap_mask_offset(daily, offset)
    nonoverlap = daily[mask]
    if nonoverlap.empty:
        return None

    market_daily_raw = test.groupby(level="Date")["future_3d"].mean()
    common_idx = nonoverlap.index.intersection(market_daily_raw.index)
    nonoverlap = nonoverlap.loc[common_idx]
    bm_raw = market_daily_raw.loc[common_idx]
    if nonoverlap.empty:
        return None

    long_final = (1 + nonoverlap["long"]).prod() - 1
    bm_final   = (1 + bm_raw).prod() - 1
    return long_final - bm_final, len(nonoverlap)


def run_sensitivity(
    panel: pd.DataFrame, window_index: int, min_train_days: int, test_days: int, step_days: int
) -> None:
    train, test = get_window(panel, window_index, min_train_days, test_days, step_days)
    test_start = test.index.get_level_values("Date").min().date()
    test_end   = test.index.get_level_values("Date").max().date()
    print(f"\n鎖定視窗 {window_index}：測試期 {test_start} ～ {test_end}")

    print("訓練 ML 模型...")
    model = train_model(train)
    preds_ml = model.predict(test[ALL_FEATURES])
    daily_ml = evaluate(test, preds_ml)

    preds_naive = naive_predictions(test)
    daily_naive = evaluate(test, preds_naive)

    print("\n" + "=" * 70)
    print(f"📊 視窗 {window_index} 複利抽樣路徑敏感度分析（{FORWARD_DAYS}種起始offset）")
    print("=" * 70)
    print(f"{'起始offset':<12}{'ML超額報酬':>14}{'樣本數':>8}   |   {'簡單版超額報酬':>16}{'樣本數':>8}")

    ml_results, naive_results = [], []
    for offset in range(FORWARD_DAYS):
        ml_res    = compounded_excess_for_offset(test, daily_ml, offset)
        naive_res = compounded_excess_for_offset(test, daily_naive, offset)
        ml_excess,    ml_n    = ml_res    if ml_res    is not None else (np.nan, 0)
        naive_excess, naive_n = naive_res if naive_res is not None else (np.nan, 0)
        ml_results.append(ml_excess)
        naive_results.append(naive_excess)
        print(f"{offset:<12}{ml_excess*100:>+13.1f}%{ml_n:>8}   |   {naive_excess*100:>+15.1f}%{naive_n:>8}")

    ml_arr    = np.array(ml_results, dtype=float)
    naive_arr = np.array(naive_results, dtype=float)

    ml_span    = np.nanmax(ml_arr) - np.nanmin(ml_arr)
    naive_span = np.nanmax(naive_arr) - np.nanmin(naive_arr)

    print(
        f"\nML超額報酬：平均 {np.nanmean(ml_arr)*100:+.1f}%　"
        f"範圍 [{np.nanmin(ml_arr)*100:+.1f}%, {np.nanmax(ml_arr)*100:+.1f}%]　"
        f"跨度 {ml_span*100:.1f} 個百分點"
    )
    print(
        f"簡單版超額報酬：平均 {np.nanmean(naive_arr)*100:+.1f}%　"
        f"範圍 [{np.nanmin(naive_arr)*100:+.1f}%, {np.nanmax(naive_arr)*100:+.1f}%]　"
        f"跨度 {naive_span*100:.1f} 個百分點"
    )

    # 判讀門檻：跨度超過平均值一半、且絕對值本身也夠大（>15個百分點），
    # 才視為「對起始點敏感」；這個門檻是起點，不是絕對標準，數字本身比判讀重要
    ml_mean_abs = abs(np.nanmean(ml_arr))
    if ml_span > 0.15 and ml_span > ml_mean_abs * 0.5:
        print(
            f"\n判讀：ML超額報酬對「從哪天開始抽樣」很敏感"
            f"（3個offset的跨度達 {ml_span*100:.0f} 個百分點，"
            f"相對平均值 {np.nanmean(ml_arr)*100:+.1f}% 波動很大），"
            f"這個視窗的單一數字可能有相當大的抽樣路徑雜訊，不宜過度解讀成"
            f"「這段期間選股品質真的差」"
        )
    else:
        print(
            f"\n判讀：ML超額報酬在3個offset之間相對穩定"
            f"（跨度 {ml_span*100:.0f} 個百分點），這個視窗的結果比較不像是"
            f"抽樣路徑雜訊造成的，可以較有信心地當成真實表現差異"
        )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--window-index", type=int, required=True,
                        help="要分析第幾個 walk-forward 視窗（1-based，對齊 day30/day31/day32 的印出編號）")
    parser.add_argument("--min-train-days", type=int, default=DEFAULT_MIN_TRAIN_DAYS)
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument("--step-days", type=int, default=DEFAULT_STEP_DAYS)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH,
                        help="資料快取檔位置，跟 day28/30/31/32 共用同一份才能互相比較")
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

    run_sensitivity(panel, args.window_index, args.min_train_days, args.test_days, args.step_days)


if __name__ == "__main__":
    main()
