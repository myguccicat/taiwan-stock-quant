# day31_naive_vs_model.py — 模型複雜度對照組：純 sector_momentum 規則 vs 完整 RandomForest 模型
#
# 動機：day28 的特徵重要性顯示 sector_momentum 佔 37.93%，sector_rank +
# inter_sector_rank（真正在做「產業內選股」的兩個特徵）加起來不到 2%。
# 這暗示模型的價值可能主要來自「追蹤最近漲最兇的產業」，而不是「在產業內
# 挑出真正強的個股」——如果屬實，一個不需要訓練、直接拿 sector_momentum
# 排名的簡單規則，應該就能複製掉大部分的超額報酬。
#
# 做法：用完全相同的一批下載資料、完全相同的訓練/測試切分或 walk-forward
# 視窗，一邊跑真正的 RandomForest 模型，一邊只用 sector_momentum 原始數值
# 直接排名（不訓練任何模型），兩邊套用 day28 完全相同的 evaluate() /
# compounded_returns() / benchmark_returns() 邏輯，確保比較公平。
#
# 執行：python day31_naive_vs_model.py
#   --mode single      只做單一測試期比較（跟 day28 預設切分一致，較快）
#   --mode walkforward 做 walk-forward 多視窗比較（跟 day30 預設參數一致，較慢）
#   --mode both        兩種都做（預設）

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

# 重用 day28／day30 已經驗證過的下載、特徵工程、評估邏輯，避免同一套計算
# 散落在多個檔案裡彼此漂移（例如「不重疊複利」的修正，這裡不用再修一次）
from day28_sector_ranking import (
    STOCKS,
    ALL_FEATURES,
    TOP_N,
    FORWARD_DAYS,
    DEFAULT_CACHE_PATH,
    configure_yfinance_cache,
    load_or_download_market_data,
    build_panel,
    split_train_test,
    train_model,
    evaluate,
    nonoverlap_mask,
    compounded_returns,
    benchmark_returns,
)
from day30_walkforward import generate_walkforward_windows

DEFAULT_PERIOD         = "3y"
DEFAULT_MIN_TRAIN_DAYS = 300
DEFAULT_TEST_DAYS      = 60
DEFAULT_STEP_DAYS      = 60
DEFAULT_OUTPUT         = "day31_naive_vs_model.png"
DEFAULT_CSV_OUTPUT     = "day31_walkforward_comparison.csv"

NAIVE_FEATURE = "sector_momentum"


def naive_predictions(test: pd.DataFrame) -> np.ndarray:
    """簡單版策略：不訓練任何模型，直接拿 sector_momentum 原始數值當排名依據。
    數值越高代表「這個股票所屬產業最近平均漲得越兇」，越優先做多。"""
    return test[NAIVE_FEATURE].values


# ── 單一測試期比較（對齊 day28 的預設切分） ──────────────────────────────

def run_single_period_comparison(panel: pd.DataFrame) -> dict:
    train, test = split_train_test(panel)

    print(f"訓練：{train.index.get_level_values('Date').min().date()} ～ "
          f"{train.index.get_level_values('Date').max().date()}")
    print(f"測試：{test.index.get_level_values('Date').min().date()} ～ "
          f"{test.index.get_level_values('Date').max().date()}")

    print("\n訓練 RandomForest 模型（ML版）...")
    model = train_model(train)
    preds_ml = model.predict(test[ALL_FEATURES])
    daily_ml = evaluate(test, preds_ml)
    lc_ml, sc_ml, ls_ml, nonoverlap_ml = compounded_returns(daily_ml)
    bm_ml = benchmark_returns(test, nonoverlap_ml.index)

    print("計算簡單版規則（僅用 sector_momentum 排名，不訓練模型）...")
    preds_naive = naive_predictions(test)
    daily_naive = evaluate(test, preds_naive)
    lc_naive, sc_naive, ls_naive, nonoverlap_naive = compounded_returns(daily_naive)
    bm_naive = benchmark_returns(test, nonoverlap_naive.index)

    ic_ml    = daily_ml["spearman"].mean()
    ic_naive = daily_naive["spearman"].mean()

    long_ml_final    = lc_ml.iloc[-1]    if len(lc_ml)    else np.nan
    long_naive_final = lc_naive.iloc[-1] if len(lc_naive) else np.nan
    bm_ml_final       = bm_ml.iloc[-1]     if len(bm_ml)     else np.nan
    bm_naive_final     = bm_naive.iloc[-1] if len(bm_naive) else np.nan

    excess_ml    = long_ml_final - bm_ml_final
    excess_naive = long_naive_final - bm_naive_final

    print("\n" + "=" * 60)
    print("📊 單一測試期：ML模型 vs 簡單版產業動能規則")
    print("=" * 60)
    print(f"{'指標':<20}{'ML模型':>15}{'簡單版規則':>15}")
    print(f"{'Rank IC':<20}{ic_ml:>+15.4f}{ic_naive:>+15.4f}")
    print(f"{'前' + str(TOP_N) + '名做多累積報酬':<18}{long_ml_final*100:>+14.1f}%{long_naive_final*100:>+14.1f}%")
    print(f"{'全市場基準':<20}{bm_ml_final*100:>+14.1f}%{bm_naive_final*100:>+14.1f}%")
    print(f"{'超額報酬':<20}{excess_ml*100:>+14.1f}%{excess_naive*100:>+14.1f}%")

    if excess_ml > 0:
        capture_ratio = excess_naive / excess_ml
        print(f"\n簡單版規則捕捉到 ML 模型超額報酬的比例：{capture_ratio*100:.0f}%")
    else:
        capture_ratio = np.nan
        print(f"\nML 模型本身超額報酬不是正的，「捕捉比例」這個算法在這裡沒有意義")

    return {
        "ic_ml": ic_ml, "ic_naive": ic_naive,
        "long_ml": long_ml_final, "long_naive": long_naive_final,
        "bm_ml": bm_ml_final, "bm_naive": bm_naive_final,
        "excess_ml": excess_ml, "excess_naive": excess_naive,
        "capture_ratio": capture_ratio,
        "curves_ml": (lc_ml, sc_ml, ls_ml, bm_ml),
        "curves_naive": (lc_naive, sc_naive, ls_naive, bm_naive),
    }


# ── Walk-forward 多視窗比較（對齊 day30 的預設參數） ──────────────────────

def run_walkforward_comparison(
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

    print(f"\n共產生 {len(windows)} 個 walk-forward 視窗（ML模型 vs 簡單版規則）")

    rows = []
    stitched = {k: [] for k in ("long_ml", "long_naive", "benchmark")}

    for i, (train_dates, test_dates) in enumerate(windows, start=1):
        train = clean.loc[clean.index.get_level_values("Date").isin(train_dates)]
        test  = clean.loc[clean.index.get_level_values("Date").isin(test_dates)]

        test_start, test_end = test_dates.min().date(), test_dates.max().date()
        print(f"\n[視窗 {i}/{len(windows)}] 測試 {test_start}~{test_end}（{len(test_dates)}天）")

        # ML 模型（需要訓練）
        model = train_model(train)
        preds_ml = model.predict(test[ALL_FEATURES])
        daily_ml = evaluate(test, preds_ml)

        # 簡單版規則（不需要訓練，直接用 sector_momentum）
        preds_naive = naive_predictions(test)
        daily_naive = evaluate(test, preds_naive)

        if daily_ml.empty or daily_naive.empty:
            print("   ⚠️ 這個視窗沒有足夠的樣本，跳過")
            continue

        mask_ml    = nonoverlap_mask(daily_ml)
        mask_naive = nonoverlap_mask(daily_naive)
        nonoverlap_ml    = daily_ml[mask_ml]
        nonoverlap_naive = daily_naive[mask_naive]

        market_daily_raw = test.groupby(level="Date")["future_3d"].mean()

        idx_ml = nonoverlap_ml.index.intersection(market_daily_raw.index)
        idx_naive = nonoverlap_naive.index.intersection(market_daily_raw.index)
        nonoverlap_ml    = nonoverlap_ml.loc[idx_ml]
        nonoverlap_naive = nonoverlap_naive.loc[idx_naive]
        bm_raw_ml    = market_daily_raw.loc[idx_ml]
        bm_raw_naive = market_daily_raw.loc[idx_naive]

        if nonoverlap_ml.empty or nonoverlap_naive.empty:
            print("   ⚠️ 這個視窗篩選後沒有可用的不重疊樣本，跳過")
            continue

        long_ml_final    = (1 + nonoverlap_ml["long"]).prod() - 1
        long_naive_final = (1 + nonoverlap_naive["long"]).prod() - 1
        bm_ml_final     = (1 + bm_raw_ml).prod() - 1
        bm_naive_final  = (1 + bm_raw_naive).prod() - 1

        excess_ml    = long_ml_final - bm_ml_final
        excess_naive = long_naive_final - bm_naive_final

        ic_ml    = daily_ml["spearman"].mean()
        ic_naive = daily_naive["spearman"].mean()

        print(
            f"   ML   : IC={ic_ml:+.4f}  前{TOP_N}名={long_ml_final*100:+.1f}%  超額={excess_ml*100:+.1f}%"
        )
        print(
            f"   簡單版: IC={ic_naive:+.4f}  前{TOP_N}名={long_naive_final*100:+.1f}%  超額={excess_naive*100:+.1f}%"
        )

        rows.append({
            "window":        i,
            "test_start":    test_start,
            "test_end":      test_end,
            "ic_ml":         round(ic_ml, 4),
            "ic_naive":      round(ic_naive, 4),
            "long_ml":       round(long_ml_final, 4),
            "long_naive":    round(long_naive_final, 4),
            "benchmark_ml":  round(bm_ml_final, 4),
            "benchmark_naive": round(bm_naive_final, 4),
            "excess_ml":     round(excess_ml, 4),
            "excess_naive":  round(excess_naive, 4),
        })

        stitched["long_ml"].extend(nonoverlap_ml["long"].tolist())
        stitched["long_naive"].extend(nonoverlap_naive["long"].tolist())
        # 基準用 ML 那邊的不重疊日期就好（兩邊理論上日期集合相同，這裡取 ML 版本即可）
        stitched["benchmark"].extend(bm_raw_ml.tolist())

    results_df = pd.DataFrame(rows)
    curves = {k: (1 + pd.Series(v, dtype=float)).cumprod() - 1 for k, v in stitched.items()}
    return results_df, curves


def print_walkforward_summary(results_df: pd.DataFrame, curves: dict[str, pd.Series]) -> None:
    print("\n" + "=" * 60)
    print("📊 Walk-forward：ML模型 vs 簡單版產業動能規則 總結")
    print("=" * 60)

    if results_df.empty:
        print("沒有任何視窗產生有效結果，無法總結。")
        print("=" * 60)
        return

    n = len(results_df)
    win_rate_ml    = (results_df["excess_ml"] > 0).mean() * 100
    win_rate_naive = (results_df["excess_naive"] > 0).mean() * 100

    print(f"共 {n} 個視窗")
    print(f"\n{'指標':<24}{'ML模型':>15}{'簡單版規則':>15}")
    print(f"{'平均 IC':<22}{results_df['ic_ml'].mean():>+15.4f}{results_df['ic_naive'].mean():>+15.4f}")
    print(f"{'平均超額報酬':<20}{results_df['excess_ml'].mean()*100:>+14.1f}%{results_df['excess_naive'].mean()*100:>+14.1f}%")
    print(f"{'超額報酬勝率':<20}{win_rate_ml:>14.0f}%{win_rate_naive:>14.0f}%")

    long_ml_final    = curves["long_ml"].iloc[-1]
    long_naive_final = curves["long_naive"].iloc[-1]
    bm_final          = curves["benchmark"].iloc[-1]
    excess_ml_final    = long_ml_final - bm_final
    excess_naive_final = long_naive_final - bm_final

    print(f"\n串接後樣本外累積報酬：")
    print(f"  ML模型前{TOP_N}名：{long_ml_final*100:+.1f}%")
    print(f"  簡單版前{TOP_N}名：{long_naive_final*100:+.1f}%")
    print(f"  全市場基準：{bm_final*100:+.1f}%")

    if excess_ml_final > 0:
        capture_ratio = excess_naive_final / excess_ml_final
        print(f"\n簡單版規則捕捉到 ML 模型（串接後）超額報酬的比例：{capture_ratio*100:.0f}%")

        if capture_ratio >= 0.8:
            verdict = (
                "簡單版規則就能複製掉八成以上的超額報酬，代表現在這套 11個技術指標 "
                "+ RandomForest 的複雜度，目前沒有明顯換到額外價值。可以考慮直接改用簡單規則，"
                "省下訓練與維護成本，或者把心力放在改良 sector_momentum 這個核心訊號本身"
            )
        elif capture_ratio >= 0.4:
            verdict = (
                "簡單版規則能捕捉到部分但不是全部的超額報酬，代表 ML 模型確實有貢獻，"
                "但貢獻有限。可以評估這個增量是否值得維持模型複雜度，"
                "或者嘗試簡化特徵集（例如只留 sector_momentum + 少數幾個技術指標）重新訓練看看"
            )
        else:
            verdict = (
                "簡單版規則遠遠做不到 ML 模型的表現，代表模型真的有在利用 sector_momentum "
                "以外的資訊做出額外價值，複雜度目前是有換到東西的"
            )
    else:
        verdict = "ML 模型本身在串接後的超額報酬不是正的，「捕捉比例」這個算法在這裡沒有意義，兩者都不建議直接拿去用"

    print(f"\n判讀：{verdict}")
    print("=" * 60)


def create_comparison_figure(
    single_result: dict | None,
    wf_results_df: pd.DataFrame | None,
    wf_curves: dict[str, pd.Series] | None,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("ML模型 vs 簡單版產業動能規則 對照圖", fontsize=14, fontweight="bold")

    # 左上：單一測試期累積報酬對照
    ax = axes[0, 0]
    if single_result is not None:
        lc_ml, sc_ml, ls_ml, bm_ml = single_result["curves_ml"]
        lc_naive, sc_naive, ls_naive, bm_naive = single_result["curves_naive"]
        ax.plot(lc_ml.index, lc_ml * 100, color="#7C3AED", lw=1.8, label="ML模型 前N名")
        ax.plot(lc_naive.index, lc_naive * 100, color="#F59E0B", lw=1.8, ls="--", label="簡單版規則 前N名")
        ax.plot(bm_ml.index, bm_ml * 100, color="#666666", lw=1.2, ls=":", label="全市場基準")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title("單一測試期：累積報酬對照")
        ax.set_ylabel("報酬率（%）")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "未執行單一測試期比較", ha="center", va="center")

    # 右上：Walk-forward 串接後累積報酬對照
    ax = axes[0, 1]
    if wf_curves is not None and not wf_results_df.empty:
        ax.plot(wf_curves["long_ml"].index, wf_curves["long_ml"] * 100,
                 color="#7C3AED", lw=1.8, label="ML模型 前N名")
        ax.plot(wf_curves["long_naive"].index, wf_curves["long_naive"] * 100,
                 color="#F59E0B", lw=1.8, ls="--", label="簡單版規則 前N名")
        ax.plot(wf_curves["benchmark"].index, wf_curves["benchmark"] * 100,
                 color="#666666", lw=1.2, ls=":", label="全市場基準")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title("Walk-forward：串接後累積報酬對照")
        ax.set_xlabel("不重疊換倉次數（跨視窗串接）")
        ax.set_ylabel("報酬率（%）")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "未執行 walk-forward 比較", ha="center", va="center")

    # 左下：各視窗超額報酬對照（分組長條）
    ax = axes[1, 0]
    if wf_results_df is not None and not wf_results_df.empty:
        width = 0.35
        x = np.arange(len(wf_results_df))
        ax.bar(x - width/2, wf_results_df["excess_ml"] * 100, width,
               color="#7C3AED", alpha=0.85, label="ML模型")
        ax.bar(x + width/2, wf_results_df["excess_naive"] * 100, width,
               color="#F59E0B", alpha=0.85, label="簡單版規則")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(wf_results_df["window"])
        ax.set_title("各視窗超額報酬對照")
        ax.set_xlabel("視窗編號")
        ax.set_ylabel("超額報酬（%）")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.axis("off")

    # 右下：文字摘要
    ax = axes[1, 1]
    ax.axis("off")
    lines = []
    if single_result is not None:
        cr = single_result["capture_ratio"]
        cr_str = f"{cr*100:.0f}%" if not np.isnan(cr) else "N/A（ML超額報酬非正）"
        lines.append("【單一測試期】")
        lines.append(f"ML超額：{single_result['excess_ml']*100:+.1f}%")
        lines.append(f"簡單版超額：{single_result['excess_naive']*100:+.1f}%")
        lines.append(f"捕捉比例：{cr_str}")
        lines.append("")
    if wf_results_df is not None and not wf_results_df.empty:
        long_ml_final    = wf_curves["long_ml"].iloc[-1]
        long_naive_final = wf_curves["long_naive"].iloc[-1]
        bm_final          = wf_curves["benchmark"].iloc[-1]
        excess_ml_final    = long_ml_final - bm_final
        excess_naive_final = long_naive_final - bm_final
        cr_wf = (excess_naive_final / excess_ml_final) if excess_ml_final > 0 else np.nan
        cr_wf_str = f"{cr_wf*100:.0f}%" if not np.isnan(cr_wf) else "N/A（ML超額報酬非正）"
        lines.append("【Walk-forward 串接後】")
        lines.append(f"ML超額：{excess_ml_final*100:+.1f}%")
        lines.append(f"簡單版超額：{excess_naive_final*100:+.1f}%")
        lines.append(f"捕捉比例：{cr_wf_str}")
    summary_text = "\n".join(lines) if lines else "沒有可顯示的結果"
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=12, va="top", family="Microsoft JhengHei")
    ax.set_title("總結")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n圖表已存檔：{output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--mode", choices=["single", "walkforward", "both"], default="both")
    parser.add_argument("--min-train-days", type=int, default=DEFAULT_MIN_TRAIN_DAYS)
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument("--step-days", type=int, default=DEFAULT_STEP_DAYS)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name(DEFAULT_OUTPUT))
    parser.add_argument("--csv-output", type=Path,
                        default=Path(__file__).with_name(DEFAULT_CSV_OUTPUT))
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH,
                        help="資料快取檔位置，跟 day28/30/32/33 共用同一份才能互相比較")
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

    single_result = None
    wf_results_df = None
    wf_curves = None

    if args.mode in ("single", "both"):
        single_result = run_single_period_comparison(panel)

    if args.mode in ("walkforward", "both"):
        if args.mode == "both":
            print(
                f"\n⚠️ 接下來會做 walk-forward 比較，ML那邊要重複訓練模型多次，"
                f"執行時間會再拉長，請耐心等候。"
            )
        wf_results_df, wf_curves = run_walkforward_comparison(
            panel, args.min_train_days, args.test_days, args.step_days
        )
        print_walkforward_summary(wf_results_df, wf_curves)
        if not wf_results_df.empty:
            wf_results_df.to_csv(args.csv_output, index=False, encoding="utf-8-sig")
            print(f"\n💾 各視窗對照結果已存檔：{args.csv_output}")

    create_comparison_figure(single_result, wf_results_df, wf_curves, args.output)


if __name__ == "__main__":
    main()
