# day30_walkforward.py — Walk-forward 滾動驗證
#
# 目的：day28_sector_ranking.py 只用一段連續的測試期（2026-01~2026-08，剛好是
# 單邊多頭）驗證模型，前5名超額報酬 +194% 很可能只是「這半年順風」，沒驗證過
# 模型在盤整或下跌期間還撐不撐得住。
#
# 做法：把整段歷史切成好幾個先後相接的「訓練→測試」視窗（walk-forward），
# 每個視窗都重新訓練模型、只用該視窗的測試期驗證，最後把所有視窗的樣本外
# 報酬串接起來，才是比較誠實的長期表現估計。訓練視窗採用「擴張窗」
# （expanding window：每次都用從頭到目前為止的所有資料訓練），測試視窗則是
# 往前滾動、彼此不重疊。
#
# 執行：python day30_walkforward.py
# 可選參數：--min-train-days（第一個視窗最少要多少天訓練資料，預設300）
#          --test-days（每個視窗的測試天數，預設60）
#          --step-days（每次往前滾動幾天，預設60，等於 test-days 代表測試期彼此不重疊）
#
# 注意：這支腳本會重複訓練模型好幾次（視窗數視參數而定，預設約5~7個），
# 執行時間會比 day28 久上數倍，屬正常現象。

from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

# 重用 day28 的下載、特徵工程、訓練、評估邏輯，避免同一套計算邏輯散落在兩個檔案裡
# 彼此漂移（例如上次修好的「不重疊複利」bug，這裡就不用再修一次）
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
    nonoverlap_mask,
    top_pick_frequency,
)

DEFAULT_PERIOD         = "3y"
DEFAULT_MIN_TRAIN_DAYS = 300
DEFAULT_TEST_DAYS      = 60
DEFAULT_STEP_DAYS      = 60
DEFAULT_OUTPUT         = "day30_walkforward.png"
DEFAULT_CSV_OUTPUT     = "walkforward_windows.csv"


def generate_walkforward_windows(
    dates: pd.DatetimeIndex, min_train_days: int, test_days: int, step_days: int
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """產生一系列（訓練日期, 測試日期）視窗。訓練窗是擴張窗（從頭累積到當下），
    測試窗長度固定、彼此依 step_days 往前滾動（預設等於 test_days，即不重疊）。"""
    windows = []
    train_end_idx = min_train_days
    while True:
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_days
        if test_end_idx > len(dates):
            break
        train_dates = dates[:train_end_idx]
        test_dates  = dates[test_start_idx:test_end_idx]
        windows.append((train_dates, test_dates))
        train_end_idx += step_days
    return windows


def run_walkforward(
    panel: pd.DataFrame, min_train_days: int, test_days: int, step_days: int
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    clean = panel.dropna(subset=ALL_FEATURES + ["future_3d"]).copy()
    dates = clean.index.get_level_values("Date").unique().sort_values()

    windows = generate_walkforward_windows(dates, min_train_days, test_days, step_days)
    if not windows:
        raise ValueError(
            "資料量不足以產生任何 walk-forward 視窗，"
            "請減少 --min-train-days 或 --test-days，或加長 --period"
        )

    print(f"共產生 {len(windows)} 個 walk-forward 視窗")

    window_rows = []
    stitched_long, stitched_short, stitched_ls, stitched_bm = [], [], [], []

    for i, (train_dates, test_dates) in enumerate(windows, start=1):
        train = clean.loc[clean.index.get_level_values("Date").isin(train_dates)]
        test  = clean.loc[clean.index.get_level_values("Date").isin(test_dates)]

        train_start, train_end = train_dates.min().date(), train_dates.max().date()
        test_start,  test_end  = test_dates.min().date(),  test_dates.max().date()
        print(
            f"\n[視窗 {i}/{len(windows)}] 訓練 {train_start}~{train_end}"
            f"（{len(train_dates)}天） → 測試 {test_start}~{test_end}（{len(test_dates)}天）"
        )

        model = train_model(train)
        preds = model.predict(test[ALL_FEATURES])
        daily = evaluate(test, preds)

        if daily.empty:
            print("   ⚠️ 這個視窗沒有足夠的樣本，跳過")
            continue

        mask       = nonoverlap_mask(daily)
        nonoverlap = daily[mask]

        # 全市場等權重基準（單期報酬，未累積），只取跟前5名相同的不重疊日期
        market_daily_raw = test.groupby(level="Date")["future_3d"].mean()
        common_idx = nonoverlap.index.intersection(market_daily_raw.index)
        nonoverlap = nonoverlap.loc[common_idx]
        bm_raw     = market_daily_raw.loc[common_idx]

        if nonoverlap.empty:
            print("   ⚠️ 這個視窗篩選後沒有可用的不重疊樣本，跳過")
            continue

        long_final  = (1 + nonoverlap["long"]).prod() - 1
        short_final = (1 + nonoverlap["short"]).prod() - 1
        ls_final    = (1 + nonoverlap["long_short"]).prod() - 1
        bm_final    = (1 + bm_raw).prod() - 1
        excess      = long_final - bm_final

        scored_test = test.copy()
        scored_test["predicted"] = preds
        freq   = top_pick_frequency(scored_test, TOP_N)
        n_days = freq.attrs.get("n_days", len(daily))
        top1_pct = (freq.iloc[0] / n_days * 100) if len(freq) and n_days else 0.0

        ic_mean = daily["spearman"].mean()

        print(
            f"   IC={ic_mean:+.4f}  前{TOP_N}名(純做多)={long_final*100:+.1f}%  "
            f"基準={bm_final*100:+.1f}%  超額={excess*100:+.1f}%  "
            f"集中度(最高單檔)={top1_pct:.0f}%  [對照] 後{TOP_N}名={short_final*100:+.1f}%"
        )

        window_rows.append({
            "window":              i,
            "train_start":         train_start,
            "train_end":           train_end,
            "test_start":          test_start,
            "test_end":            test_end,
            "ic_mean":             round(ic_mean, 4),
            "long_return":         round(long_final, 4),
            "short_return":        round(short_final, 4),
            "long_short_return":   round(ls_final, 4),
            "benchmark_return":    round(bm_final, 4),
            "excess_return":       round(excess, 4),
            "top1_freq_pct":       round(top1_pct, 1),
            "n_nonoverlap_periods": len(nonoverlap),
        })

        stitched_long.extend(nonoverlap["long"].tolist())
        stitched_short.extend(nonoverlap["short"].tolist())
        stitched_ls.extend(nonoverlap["long_short"].tolist())
        stitched_bm.extend(bm_raw.tolist())

    results_df = pd.DataFrame(window_rows)
    stitched = {
        "long":       pd.Series(stitched_long, dtype=float),
        "short":      pd.Series(stitched_short, dtype=float),
        "long_short": pd.Series(stitched_ls, dtype=float),
        "benchmark":  pd.Series(stitched_bm, dtype=float),
    }
    return results_df, stitched


def build_stitched_curve(stitched: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """把每個視窗的不重疊單期報酬依時間順序接起來，做一條連續的樣本外複利曲線。
    因為視窗本身按時間先後排列、彼此不重疊，串接後仍然是誠實的樣本外表現，
    不會有 day28 一開始那種重疊複利膨脹的問題。"""
    curves = {}
    for key, s in stitched.items():
        curves[key] = (1 + s.reset_index(drop=True)).cumprod() - 1
    return curves


def print_overall_summary(results_df: pd.DataFrame, curves: dict[str, pd.Series]) -> None:
    print("\n" + "=" * 60)
    print("📊 Walk-forward 滾動驗證總結")
    print("=" * 60)
    print(f"策略設計：純做多前{TOP_N}名（不放空後{TOP_N}名）")

    if results_df.empty:
        print("沒有任何視窗產生有效結果，無法總結。")
        print("=" * 60)
        return

    n = len(results_df)
    # excess_return = 前N名做多 - 全市場基準，本來就是「純做多策略」的超額報酬，
    # 勝率統計的一直都是這個純做多口徑，跟是否放空無關
    win_rate = (results_df["excess_return"] > 0).mean() * 100

    print(f"共 {n} 個視窗")
    print(f"平均 IC：{results_df['ic_mean'].mean():+.4f}（標準差 {results_df['ic_mean'].std():.4f}）")
    print()
    print(f"【純做多前{TOP_N}名（實際策略）】")
    print(f"平均超額報酬（相對全市場基準）：{results_df['excess_return'].mean()*100:+.1f}%")
    print(
        f"超額報酬為正的視窗比例（勝率）：{win_rate:.0f}%"
        f"（{int((results_df['excess_return'] > 0).sum())}/{n}）"
    )
    print(f"串接後的樣本外累積報酬（把每個視窗的測試期報酬依時間接起來，不是單一視窗誇大出來的數字）：")
    print(f"  前{TOP_N}名做多（實際策略）：{curves['long'].iloc[-1]*100:+.1f}%")
    print(f"  全市場基準：{curves['benchmark'].iloc[-1]*100:+.1f}%")

    print(f"\n【以下僅供研究對照，非建議操作】")
    print(f"  後{TOP_N}名（避開參考，非放空）：{curves['short'].iloc[-1]*100:+.1f}%")
    print(f"  多空組合（前做多+後放空，僅供比較）：{curves['long_short'].iloc[-1]*100:+.1f}%")

    if win_rate >= 70:
        verdict = "純做多前N名策略多數視窗都跑贏基準，訊號在不同市場區間相對穩定，可以考慮繼續往下驗證（例如小額實單測試）"
    elif win_rate >= 50:
        verdict = "純做多前N名策略勝率剛過半，訊號可能存在但不夠穩定，建議先縮小部位規模或延長觀察期，不建議直接重倉"
    else:
        verdict = "純做多前N名策略多數視窗都沒跑贏基準，先前看到的高報酬很可能只是特定市場區間（單邊多頭）的偶然結果，不建議直接拿這個模型去實戰"
    print(f"\n判讀：{verdict}")
    print("=" * 60)


def create_walkforward_figure(
    results_df: pd.DataFrame, curves: dict[str, pd.Series], output_path: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Walk-forward 滾動驗證診斷圖", fontsize=14, fontweight="bold")

    # 左上：串接後的樣本外累積報酬（純做多策略為主，放空/多空僅供對照）
    ax = axes[0, 0]
    ax.plot(curves["long"].index, curves["long"] * 100,
             color="#16A34A", lw=2.0, label=f"前{TOP_N}名做多（實際策略）")
    ax.plot(curves["benchmark"].index, curves["benchmark"] * 100,
             color="#666666", lw=1.4, ls=":", label="全市場基準")
    ax.plot(curves["short"].index, curves["short"] * 100,
             color="#DC2626", lw=1.0, ls="--", alpha=0.6, label=f"後{TOP_N}名（對照，非放空）")
    ax.plot(curves["long_short"].index, curves["long_short"] * 100,
             color="#7C3AED", lw=1.0, ls="--", alpha=0.6, label="多空組合（對照）")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("視窗串接後的樣本外累積報酬（純做多策略 vs 基準）")
    ax.set_xlabel("不重疊換倉次數（跨視窗串接）")
    ax.set_ylabel("報酬率（%）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    if not results_df.empty:
        # 右上：各視窗 IC
        ax = axes[0, 1]
        colors_ic = ["#16A34A" if v > 0 else "#DC2626" for v in results_df["ic_mean"]]
        ax.bar(results_df["window"], results_df["ic_mean"], color=colors_ic, alpha=0.85)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title("各視窗 Rank IC")
        ax.set_xlabel("視窗編號")
        ax.set_ylabel("IC")
        ax.grid(axis="y", alpha=0.3)

        # 左下：各視窗超額報酬
        ax = axes[1, 0]
        colors_ex = ["#16A34A" if v > 0 else "#DC2626" for v in results_df["excess_return"]]
        ax.bar(results_df["window"], results_df["excess_return"] * 100, color=colors_ex, alpha=0.85)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"各視窗超額報酬（純做多前{TOP_N}名 - 全市場基準）")
        ax.set_xlabel("視窗編號")
        ax.set_ylabel("超額報酬（%）")
        ax.grid(axis="y", alpha=0.3)

        # 右下：文字摘要
        ax = axes[1, 1]
        ax.axis("off")
        win_rate = (results_df["excess_return"] > 0).mean() * 100
        summary_text = (
            f"視窗數：{len(results_df)}\n\n"
            f"平均 IC：{results_df['ic_mean'].mean():+.4f}\n"
            f"IC 標準差：{results_df['ic_mean'].std():.4f}\n\n"
            f"平均超額報酬：{results_df['excess_return'].mean()*100:+.1f}%\n"
            f"超額報酬勝率：{win_rate:.0f}%\n\n"
            f"測試期涵蓋：\n{results_df['test_start'].min()} ～\n{results_df['test_end'].max()}"
        )
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=12, va="top", family="Microsoft JhengHei")
        ax.set_title("總結")
    else:
        for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"圖表已存檔：{output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--min-train-days", type=int, default=DEFAULT_MIN_TRAIN_DAYS,
                        help="第一個視窗最少要多少天訓練資料")
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS,
                        help="每個視窗的測試天數")
    parser.add_argument("--step-days", type=int, default=DEFAULT_STEP_DAYS,
                        help="每次往前滾動幾天（預設等於 test-days，即測試期彼此不重疊）")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name(DEFAULT_OUTPUT))
    parser.add_argument("--csv-output", type=Path,
                        default=Path(__file__).with_name(DEFAULT_CSV_OUTPUT))
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH,
                        help="資料快取檔位置，跟 day28/31/32/33 共用同一份才能互相比較")
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

    print(
        f"\n⚠️ 這支腳本會重複訓練模型多次（視窗數視資料量而定），"
        f"執行時間會比 day28 久上數倍，請耐心等候。"
    )

    results_df, stitched = run_walkforward(
        panel, args.min_train_days, args.test_days, args.step_days
    )
    curves = build_stitched_curve(stitched)

    print_overall_summary(results_df, curves)

    if not results_df.empty:
        results_df.to_csv(args.csv_output, index=False, encoding="utf-8-sig")
        print(f"\n💾 各視窗結果已存檔：{args.csv_output}")

    create_walkforward_figure(results_df, curves, args.output)


if __name__ == "__main__":
    main()
