# day13_report.py — 一鍵產生完整量化分析報告

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import date

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

# ══════════════════════════════════════════
#  設定區：只需改這裡
# ══════════════════════════════════════════
TARGET   = "2317.TW"   # 主要分析標的
NAME     = "鴻海"
BENCHMARK = "0050.TW"  # 基準指數
BM_NAME  = "台灣50"
PERIOD   = "1y"
MA_SHORT = 20
MA_LONG  = 60

WATCHLIST = {
    "2330.TW":"台積電","2317.TW":"鴻海","2454.TW":"聯發科",
    "2308.TW":"台達電","2382.TW":"廣達","2412.TW":"中華電",
    "2882.TW":"國泰金","1301.TW":"台塑","2357.TW":"華碩","2303.TW":"聯電",
}
# ══════════════════════════════════════════

def get_metrics(prices, ma_s=20, ma_l=60):
    df = pd.DataFrame({"c": prices})
    df["r"]   = df["c"].pct_change()
    df["ma_s"] = df["c"].rolling(ma_s).mean()
    df["ma_l"] = df["c"].rolling(ma_l).mean()
    df["sig"]  = (df["ma_s"] > df["ma_l"]).astype(int)
    df["sr"]   = df["r"] * df["sig"].shift(1)
    df = df.dropna()

    cum  = (1 + df["sr"]).cumprod()
    mkt  = (1 + df["r"]).cumprod()
    std  = df["sr"].std()

    return {
        "ret":      round((cum.iloc[-1]-1)*100, 1),
        "mkt_ret":  round((mkt.iloc[-1]-1)*100, 1),
        "excess":   round((cum.iloc[-1]-mkt.iloc[-1])*100, 1),
        "sharpe":   round(df["sr"].mean()/std*np.sqrt(252) if std>0 else 0, 2),
        "max_dd":   round(((cum-(cum.cummax()))/(cum.cummax())).min()*100, 1),
        "vol":      round(df["r"].std()*np.sqrt(252)*100, 1),
        "win_rate": round((df["sr"]>0).sum()/(df["sr"]!=0).sum()*100, 1),
        "skew":     round(df["r"].skew(), 2),
        "kurt":     round(df["r"].kurt(), 2),
        "_cum": cum, "_mkt": mkt, "_df": df,
    }

# ── 下載資料 ──────────────────────────────
print("下載資料中...")
codes = list(WATCHLIST.keys()) + [BENCHMARK]
raw   = yf.download(codes, period=PERIOD, progress=False)["Close"]
raw.columns = list(WATCHLIST.values()) + [BM_NAME]
raw   = raw.dropna()

target_prices = raw[NAME]
bm_prices     = raw[BM_NAME]
m   = get_metrics(target_prices, MA_SHORT, MA_LONG)
bm  = get_metrics(bm_prices,     MA_SHORT, MA_LONG)

# 全市場掃描
scan = {n: get_metrics(raw[n]) for n in WATCHLIST.values()}
df_scan = pd.DataFrame({
    n: {"策略報酬": v["ret"], "Sharpe": v["sharpe"],
        "最大回撤": v["max_dd"], "勝率": v["win_rate"]}
    for n, v in scan.items()
}).T.sort_values("Sharpe", ascending=False)

# ── 印出報告 ──────────────────────────────
print(f"""
╔══════════════════════════════════════════╗
║   量化分析報告  {date.today()}          ║
╠══════════════════════════════════════════╣
║ 標的：{NAME:<8}  基準：{BM_NAME}
╠══════════════════════════════════════════╣
║ 【策略績效】
║   策略累積報酬：{m['ret']:>+7.1f}%
║   市場買入持有：{m['mkt_ret']:>+7.1f}%
║   超額報酬    ：{m['excess']:>+7.1f}%
║   Sharpe Ratio：{m['sharpe']:>7.2f}
║   最大回撤    ：{m['max_dd']:>7.1f}%
║   年化波動率  ：{m['vol']:>7.1f}%
║   勝率        ：{m['win_rate']:>7.1f}%
╠══════════════════════════════════════════╣
║ 【報酬率特徵】
║   偏度：{m['skew']:>+.2f}  峰度：{m['kurt']:>.2f}
║   {'右偏（大漲次數多）' if m['skew']>0 else '左偏（大跌影響大）'}
║   {'厚尾（極端事件偏多）' if m['kurt']>0 else '薄尾（接近常態）'}
╠══════════════════════════════════════════╣
║ 【vs 基準 {BM_NAME}】
║   基準策略報酬：{bm['ret']:>+7.1f}%
║   超越基準    ：{'✓ 是' if m['ret']>bm['ret'] else '✗ 否'}
╚══════════════════════════════════════════╝
""")

print("── 全市場 Sharpe 排行 ──")
for name, row in df_scan.iterrows():
    bar = "█" * min(int(max(row["Sharpe"],0) * 8), 30)
    print(f"  {name:<5} Sharpe:{row['Sharpe']:>5.2f}  {bar}")

# ── 畫圖 ──────────────────────────────────
df_ = m["_df"]
fig = plt.figure(figsize=(15, 10))
fig.suptitle(f"{NAME} 完整量化分析報告  {date.today()}",
             fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig)

# 左上大圖：股價+均線+策略淨值
ax1 = fig.add_subplot(gs[0:2, 0:2])
ax1b = ax1.twinx()
ax1.plot(df_.index, df_["c"],    color="#94A3B8", lw=1,   label="收盤價", alpha=0.7)
ax1.plot(df_.index, df_["ma_s"], color="#F59E0B", lw=1.2, label=f"MA{MA_SHORT}", ls="--")
ax1.plot(df_.index, df_["ma_l"], color="#EF4444", lw=1.2, label=f"MA{MA_LONG}",  ls="--")
cum_pct = m["_cum"] * 100 - 100
bm_pct  = bm["_mkt"] * 100 - 100
ax1b.plot(df_.index, cum_pct,                   color="#7C3AED", lw=1.5, label="策略淨值")
ax1b.plot(bm["_mkt"].index, bm_pct.values,      color="#1D9E75", lw=1,   label=f"{BM_NAME}", ls=":")
ax1b.axhline(0, color="gray", lw=0.5, ls=":")
ax1.set_ylabel("股價（元）")
ax1b.set_ylabel("累積報酬（%）")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="upper left")
ax1.set_title(f"{NAME} 股價（還原除權息）+ 策略淨值 vs {BM_NAME}")
ax1.grid(alpha=0.3)

# 右上：績效指標卡
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis("off")
metrics_text = [
    ("策略報酬",  f"{m['ret']:+.1f}%"),
    ("市場報酬",  f"{m['mkt_ret']:+.1f}%"),
    ("超額報酬",  f"{m['excess']:+.1f}%"),
    ("Sharpe",   f"{m['sharpe']:.2f}"),
    ("最大回撤",  f"{m['max_dd']:.1f}%"),
    ("勝率",      f"{m['win_rate']:.1f}%"),
]
for i, (label, val) in enumerate(metrics_text):
    color = "#16A34A" if (i==2 and m["excess"]>0) or \
                         (i==3 and m["sharpe"]>1) else \
            "#DC2626" if (i==2 and m["excess"]<0) else "#1E293B"
    ax2.text(0.05, 0.92-i*0.15, label,
             transform=ax2.transAxes, fontsize=9,
             color="#64748B")
    ax2.text(0.95, 0.92-i*0.15, val,
             transform=ax2.transAxes, fontsize=11,
             fontweight="bold", color=color, ha="right")
ax2.set_title("績效摘要", fontsize=10)
ax2.add_patch(plt.Rectangle((0,0),1,1, fill=False,
              edgecolor="#E2E8F0", lw=1,
              transform=ax2.transAxes))

# 中右：報酬率分佈
ax3 = fig.add_subplot(gs[1, 2])
r = df_["r"] * 100
ax3.hist(r, bins=40, color="#7C3AED", alpha=0.6, density=True)
from scipy import stats
x = np.linspace(r.min(), r.max(), 200)
ax3.plot(x, stats.norm.pdf(x, r.mean(), r.std()),
         color="#EF4444", lw=1.5, ls="--", label="常態分佈")
ax3.axvline(0, color="gray", lw=0.8)
ax3.set_title("日報酬分佈", fontsize=9)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# 下方：Sharpe 排行橫條圖
ax4 = fig.add_subplot(gs[2, :])
names_sorted  = df_scan.index.tolist()
sharpe_sorted = df_scan["Sharpe"].tolist()
colors = ["#7C3AED" if n == NAME else
          "#1D9E75" if s > 2 else
          "#F59E0B" if s > 1 else "#EF4444"
          for n, s in zip(names_sorted, sharpe_sorted)]
bars = ax4.barh(names_sorted, sharpe_sorted, color=colors, alpha=0.8)
ax4.axvline(1, color="gray", lw=1, ls="--", label="Sharpe=1（基準線）")
ax4.axvline(2, color="#1D9E75", lw=1, ls="--", label="Sharpe=2（優秀）")
for bar, val in zip(bars, sharpe_sorted):
    ax4.text(val + 0.05, bar.get_y() + bar.get_height()/2,
             f"{val:.2f}", va="center", fontsize=9)
ax4.set_title("10 檔股票 Sharpe Ratio 排行", fontsize=10)
ax4.legend(fontsize=8)
ax4.grid(axis="x", alpha=0.3)
ax4.set_xlabel("Sharpe Ratio")

plt.tight_layout()
import os

# 取得目前程式碼所在的資料夾路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(current_dir, f"report_{NAME}_{date.today()}.png")

plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n報告已存檔：{fname}")