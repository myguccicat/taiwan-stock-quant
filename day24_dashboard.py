# day24_dashboard.py — 量化分析網頁介面

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from watchlist import ALL_STOCKS
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="台股量化分析系統",
    page_icon="📈",
    layout="wide"
)

# ── 側邊欄設定 ────────────────────────────────
st.sidebar.title("⚙️ 系統設定")
period   = st.sidebar.selectbox("資料期間", ["1y","2y","3y"], index=1)
top_n    = st.sidebar.slider("選股數量（Top N）", 3, 10, 5)
rebal    = st.sidebar.slider("換倉週期（天）", 3, 10, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("**自選股清單**")
st.sidebar.markdown(f"共 {len(ALL_STOCKS)} 檔")

# ── 主頁標題 ──────────────────────────────────
st.title("📈 台股量化分析系統")
st.markdown(f"資料期間：**{period}**　選股數：**Top {top_n}**　換倉週期：**{rebal} 天**")

# ── 功能頁籤 ──────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 今日訊號", "📊 個股分析", "🔄 策略回測", "📋 自選股總覽"
])

# ════════════════════════════════════════════
# TAB 1：今日訊號
# ════════════════════════════════════════════
with tab1:
    st.header("🎯 今日 ML 模型訊號")

    @st.cache_data(ttl=3600)
    def get_signal(period, top_n):
        def calc_rsi(s, n=14):
            d = s.diff()
            g = d.clip(lower=0).rolling(n).mean()
            l = (-d.clip(upper=0)).rolling(n).mean()
            return 100 - 100 / (1 + g / l)

        def build_features(price, volume):
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

        FEATURES = ["r1","r5","r20","ma5_ratio","ma20_ratio","ma60_ratio",
                    "vol_ratio","vol_5d","rsi14","bb_pos","near_high"]

        raw     = yf.download(list(ALL_STOCKS.keys()), period=period,
                              progress=False, auto_adjust=True)
        prices  = raw["Close"].rename(columns=ALL_STOCKS).ffill().dropna(axis=1, thresh=50)
        volumes = raw["Volume"].rename(columns=ALL_STOCKS).ffill()
        prices, volumes = prices.align(volumes[prices.columns], join="inner")

        frames = []
        for name in prices.columns:
            f = build_features(prices[name], volumes[name])
            f["future_5d"] = prices[name].pct_change(5).shift(-5)
            f["stock"] = name
            frames.append(f)
        panel = pd.concat(frames).dropna(subset=FEATURES + ["future_5d"])

        model = RandomForestRegressor(
            n_estimators=200, max_depth=6,
            min_samples_leaf=20, random_state=42, n_jobs=-1
        )
        model.fit(panel[FEATURES], panel["future_5d"])

        latest_date = panel.dropna(subset=FEATURES).index.max()
        latest = panel.loc[panel.index == latest_date].dropna(subset=FEATURES).copy()
        latest["score"] = model.predict(latest[FEATURES])
        ranking = latest.sort_values("score", ascending=False)

        # 今日漲跌
        raw2 = yf.download(list(ALL_STOCKS.keys()), period="2d",
                           progress=False, auto_adjust=True)["Close"]
        raw2.columns = [ALL_STOCKS.get(c, c) for c in raw2.columns]
        today_chg = ((raw2.iloc[-1] - raw2.iloc[-2]) / raw2.iloc[-2] * 100).round(2)

        return ranking, today_chg, latest_date, prices

    with st.spinner("模型訓練中，約需 30 秒..."):
        ranking, today_chg, latest_date, prices = get_signal(period, top_n)

    st.success(f"資料日期：{latest_date.date()}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟢 模型看多（前5名）")
        top5 = ranking.head(top_n)
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            name = row["stock"]
            chg  = today_chg.get(name, float("nan"))
            chg_str = f"{chg:+.2f}%" if not pd.isna(chg) else "—"
            color = "🟢" if not pd.isna(chg) and chg >= 0 else "🔴"
            st.markdown(f"**{i}. {name}** &nbsp; 分數：`{row['score']:+.5f}` &nbsp; 今日：{color} {chg_str}")

    with col2:
        st.subheader("🔴 模型看空（後5名）")
        bot5 = ranking.tail(top_n).sort_values("score")
        for i, (_, row) in enumerate(bot5.iterrows(), 1):
            name = row["stock"]
            chg  = today_chg.get(name, float("nan"))
            chg_str = f"{chg:+.2f}%" if not pd.isna(chg) else "—"
            color = "🟢" if not pd.isna(chg) and chg >= 0 else "🔴"
            st.markdown(f"**{i}. {name}** &nbsp; 分數：`{row['score']:+.5f}` &nbsp; 今日：{color} {chg_str}")

# ════════════════════════════════════════════
# TAB 2：個股分析
# ════════════════════════════════════════════
with tab2:
    st.header("📊 個股技術分析")
    selected_stock = st.selectbox("選擇股票", list(ALL_STOCKS.values()))

    @st.cache_data(ttl=3600)
    def get_stock_data(name, period):
        code = [k for k, v in ALL_STOCKS.items() if v == name][0]
        df = yf.download(code, period=period,
                         progress=False, auto_adjust=True)
        df = df[["Close","Volume"]].copy()
        df.columns = ["close","volume"]
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma60"] = df["close"].rolling(60).mean()
        df["ret"]  = df["close"].pct_change()
        df["cum"]  = (1 + df["ret"]).cumprod() - 1
        return df

    df = get_stock_data(selected_stock, period)

    # 指標卡
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("今日收盤", f"{df['close'].iloc[-1]:.1f}")
    ret_1d = df["ret"].iloc[-1] * 100
    c2.metric("今日漲跌", f"{ret_1d:+.2f}%")
    c3.metric("累積報酬", f"{df['cum'].iloc[-1]*100:+.1f}%")
    c4.metric("年化波動率", f"{df['ret'].std()*np.sqrt(252)*100:.1f}%")

    # 圖表
    fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                              gridspec_kw={"height_ratios":[3,1]})
    axes[0].plot(df.index, df["close"], color="#2563EB", lw=1.2, label="收盤價")
    axes[0].plot(df.index, df["ma20"],  color="#F59E0B", lw=1, ls="--", label="MA20")
    axes[0].plot(df.index, df["ma60"],  color="#EF4444", lw=1, ls="--", label="MA60")
    axes[0].set_ylabel("股價（元）")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_title(f"{selected_stock} 股價走勢")

    colors = ["#16A34A" if r >= 0 else "#DC2626"
              for r in df["ret"].fillna(0)]
    axes[1].bar(df.index, df["volume"], color=colors, alpha=0.7, width=1)
    axes[1].set_ylabel("成交量")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════
# TAB 3：策略回測
# ════════════════════════════════════════════
with tab3:
    st.header("🔄 ML 策略回測")
    st.info("使用左側設定調整參數，點擊下方按鈕執行回測。")

    if st.button("🚀 執行回測", type="primary"):
        with st.spinner("回測執行中..."):

            @st.cache_data(ttl=1800)
            def run_backtest(period, top_n, rebal):
                def calc_rsi(s, n=14):
                    d = s.diff()
                    g = d.clip(lower=0).rolling(n).mean()
                    l = (-d.clip(upper=0)).rolling(n).mean()
                    return 100 - 100 / (1 + g / l)

                def build_feat(price, volume):
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

                FEATURES = ["r1","r5","r20","ma5_ratio","ma20_ratio","ma60_ratio",
                            "vol_ratio","vol_5d","rsi14","bb_pos","near_high"]
                COST = 0.001425 + 0.003

                raw     = yf.download(list(ALL_STOCKS.keys()), period=period,
                                      progress=False, auto_adjust=True)
                prices  = raw["Close"].rename(columns=ALL_STOCKS).ffill().dropna(axis=1, thresh=200)
                volumes = raw["Volume"].rename(columns=ALL_STOCKS).ffill()
                prices, volumes = prices.align(volumes[prices.columns], join="inner")
                names   = list(prices.columns)

                frames = []
                for name in names:
                    f = build_feat(prices[name], volumes[name])
                    f["stock"] = name
                    frames.append(f)
                panel = pd.concat(frames).dropna()

                dates      = sorted(panel.index.unique())
                split_date = dates[int(len(dates) * 0.7)]
                train = panel[panel.index <= split_date]
                test  = panel[panel.index >  split_date]

                model = RandomForestRegressor(
                    n_estimators=300, max_depth=6,
                    min_samples_leaf=20, random_state=42, n_jobs=-1
                )
                model.fit(train[FEATURES], train["future_5d"])
                test = test.copy()
                test["pred"] = model.predict(test[FEATURES])

                rebal_dates = [d for i, d in enumerate(sorted(test.index.unique()))
                               if i % rebal == 0]
                results, prev = [], []
                for i, date in enumerate(rebal_dates[:-1]):
                    next_date = rebal_dates[i + 1]
                    day_data  = test.loc[date].sort_values("pred", ascending=False)
                    selected  = day_data["stock"].iloc[:top_n].tolist()
                    rets = []
                    for s in selected:
                        try:
                            rets.append((prices[s].loc[next_date] - prices[s].loc[date])
                                        / prices[s].loc[date])
                        except:
                            rets.append(0.0)
                    avg_ret  = np.mean(rets)
                    turnover = len(set(selected) - set(prev)) / top_n
                    cost     = turnover * COST
                    results.append({"date": date, "return": avg_ret - cost,
                                    "holdings": selected})
                    prev = selected

                res    = pd.DataFrame(results).set_index("date")
                ml_cum = (1 + res["return"]).cumprod() - 1

                bm_rets = []
                for i, date in enumerate(rebal_dates[:-1]):
                    next_date = rebal_dates[i + 1]
                    day_rets  = [(prices[s].loc[next_date] - prices[s].loc[date])
                                 / prices[s].loc[date]
                                 for s in names
                                 if date in prices[s].index and next_date in prices[s].index]
                    bm_rets.append({"date": date, "return": np.mean(day_rets)})
                bm     = pd.DataFrame(bm_rets).set_index("date")
                bm_cum = (1 + bm["return"]).cumprod() - 1

                return res, ml_cum, bm_cum, bm, split_date

            res, ml_cum, bm_cum, bm, split_date = run_backtest(period, top_n, rebal)

            ml_annual = (1 + res["return"].mean()) ** (252 / rebal) - 1
            bm_annual = (1 + bm["return"].mean()) ** (252 / rebal) - 1
            sharpe    = res["return"].mean() / res["return"].std() * np.sqrt(252 / rebal)
            maxdd     = ((ml_cum + 1) / (ml_cum + 1).cummax() - 1).min()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ML策略年化報酬", f"{ml_annual*100:+.1f}%")
            c2.metric("超額報酬", f"{(ml_annual-bm_annual)*100:+.1f}%")
            c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
            c4.metric("最大回撤", f"{maxdd*100:.1f}%")

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(ml_cum.index, ml_cum*100, color="#7C3AED",
                         lw=1.5, label=f"ML策略 {ml_cum.iloc[-1]*100:+.1f}%")
            axes[0].plot(bm_cum.index, bm_cum*100, color="#94A3B8",
                         lw=1.2, ls="--", label=f"等權基準 {bm_cum.iloc[-1]*100:+.1f}%")
            axes[0].axhline(0, color="gray", lw=0.5)
            axes[0].set_title("累積報酬對比")
            axes[0].legend(fontsize=9)
            axes[0].grid(alpha=0.3)

            alpha = ml_cum - bm_cum.reindex(ml_cum.index).ffill()
            axes[1].plot(alpha.index, alpha*100, color="#1D9E75", lw=1.5)
            axes[1].axhline(0, color="gray", lw=0.8, ls="--")
            axes[1].fill_between(alpha.index, alpha*100, 0,
                                 where=alpha >= 0, alpha=0.15, color="#1D9E75")
            axes[1].fill_between(alpha.index, alpha*100, 0,
                                 where=alpha < 0, alpha=0.15, color="#EF4444")
            axes[1].set_title("累積超額報酬（Alpha）")
            axes[1].grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ════════════════════════════════════════════
# TAB 4：自選股總覽
# ════════════════════════════════════════════
with tab4:
    st.header("📋 自選股即時總覽")

    @st.cache_data(ttl=300)
    def get_all_quotes():
        raw = yf.download(list(ALL_STOCKS.keys()), period="2d",
                          progress=False, auto_adjust=True)["Close"]
        raw.columns = [ALL_STOCKS.get(c, c) for c in raw.columns]
        raw = raw.dropna(axis=1)
        chg    = ((raw.iloc[-1] - raw.iloc[-2]) / raw.iloc[-2] * 100).round(2)
        price  = raw.iloc[-1].round(1)
        df     = pd.DataFrame({"收盤價": price, "漲跌幅(%)": chg})
        return df.sort_values("漲跌幅(%)", ascending=False)

    with st.spinner("載入報價..."):
        quote_df = get_all_quotes()

    up   = (quote_df["漲跌幅(%)"] > 0).sum()
    down = (quote_df["漲跌幅(%)"] < 0).sum()
    avg  = quote_df["漲跌幅(%)"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("上漲", f"{up} 檔")
    c2.metric("下跌", f"{down} 檔")
    c3.metric("平均漲跌幅", f"{avg:+.2f}%")

    def color_pct(val):
        color = "#085041" if val > 0 else "#A32D2D" if val < 0 else "gray"
        return f"color: {color}; font-weight: bold"

    st.dataframe(
        quote_df.style.applymap(color_pct, subset=["漲跌幅(%)"]),
        use_container_width=True,
        height=600,
    )