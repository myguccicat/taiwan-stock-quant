# daily_run.py — 每日自動執行腳本

import os
import sys
import re
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, datetime
from sklearn.ensemble import RandomForestRegressor
from watchlist import ALL_STOCKS
import warnings
from send_market_briefing_line import send_line
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE  = os.path.join(BASE_DIR, "daily_log.txt")
SIGNAL_FILE = os.path.join(BASE_DIR, "today_signal.txt")

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

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
    return df

FEATURES = ["r1","r5","r20","ma5_ratio","ma20_ratio","ma60_ratio",
            "vol_ratio","vol_5d","rsi14","bb_pos","near_high"]

def sync_holdings_from_sinopac():
    """從永豐金 API 同步庫存，只更新 watchlist.py 的 MY_HOLDINGS 區塊"""
    MIN_STOCKS = 5
    OTC_CODES = {
        "3317","6643","8277","3131","8027","7556",
        "4971","3105","6953","3680"
    }

    def to_yf_code(code):
        return f"{code}.TWO" if code in OTC_CODES else f"{code}.TW"

    # 從 sj-trading 資料夾讀取 config
    sj_path = r"C:\Users\user\Desktop\sj-trading"
    sys.path.insert(0, sj_path)
    from config import SHIOAJI_API_KEY, SHIOAJI_SECRET_KEY
    import shioaji as sj

    api = sj.Shioaji(simulation=False)
    api.login(api_key=SHIOAJI_API_KEY, secret_key=SHIOAJI_SECRET_KEY)

    positions = api.list_positions(api.stock_account, unit=sj.constant.Unit.Share)
    new_holdings = {}
    for pos in positions:
        yf_code = to_yf_code(pos.code)
        try:
            contract = api.Contracts.Stocks[pos.code]
            name = contract.name if contract else pos.code
        except:
            name = pos.code
        new_holdings[yf_code] = name

    api.logout()

    if len(new_holdings) < MIN_STOCKS:
        raise ValueError(f"抓到 {len(new_holdings)} 檔，低於最低門檻 {MIN_STOCKS} 檔，不覆蓋")

    # 更新 watchlist.py 的 MY_HOLDINGS 區塊
    watchlist_path = os.path.join(BASE_DIR, "watchlist.py")
    original = open(watchlist_path, encoding="utf-8").read()

    holdings_lines = ["MY_HOLDINGS = {\n"]
    for code, name in sorted(new_holdings.items()):
        otc = "  # OTC" if ".TWO" in code else ""
        holdings_lines.append(f'    "{code}":  "{name}",{otc}\n')
    holdings_lines.append("}\n")
    new_block = "".join(holdings_lines)

    pattern = r"MY_HOLDINGS = \{[^}]*\}\n"
    if not re.search(pattern, original, re.DOTALL):
        raise ValueError("找不到 MY_HOLDINGS 區塊，請確認 watchlist.py 格式")

    new_content = re.sub(pattern, new_block, original, flags=re.DOTALL)
    open(watchlist_path, "w", encoding="utf-8").write(new_content)

    return len(new_holdings)

def main():
    today = date.today()
    log(f"===== 每日自動執行開始 {today} =====")

    # ── 0. 同步永豐金庫存 ─────────────────────────
    log("嘗試同步永豐金庫存...")
    try:
        n = sync_holdings_from_sinopac()
        # 重新載入更新後的 watchlist
        import importlib
        import watchlist as wl
        importlib.reload(wl)
        from watchlist import ALL_STOCKS as ALL_STOCKS_NEW
        globals()["ALL_STOCKS"] = ALL_STOCKS_NEW
        log(f"庫存同步完成：{n} 檔現股")
    except Exception as e:
        log(f"庫存同步失敗，使用現有 watchlist：{e}")

    # ── 1. 下載資料 ───────────────────────────
    log("下載資料中...")
    try:
        raw     = yf.download(list(ALL_STOCKS.keys()),
                               period="1y", progress=False, auto_adjust=True)
        prices  = raw.xs("Close",  axis=1, level=0).rename(columns=ALL_STOCKS).ffill().dropna(axis=1, thresh=50)
        volumes = raw.xs("Volume", axis=1, level=0).rename(columns=ALL_STOCKS).ffill()
        prices, volumes = prices.align(volumes[prices.columns], join="inner")
        names = list(prices.columns)
        log(f"下載完成：{len(names)} 檔，{len(prices)} 個交易日")
    except Exception as e:
        log(f"下載失敗：{e}")
        sys.exit(1)

    # ── 2. 特徵工程 ───────────────────────────
    frames = []
    for name in names:
        f = build_features(prices[name], volumes[name])
        f["stock"] = name
        frames.append(f)
    panel = pd.concat(frames)

    # 訓練集：最近一年全部資料
    train = panel.dropna(subset=FEATURES)
    train = train.copy()
    train["future_5d"] = None
    for name in names:
        mask = panel["stock"] == name
        prices_s = prices[name]
        future = prices_s.pct_change(5).shift(-5)
        panel.loc[mask, "future_5d"] = future.values
    train = panel.dropna(subset=FEATURES + ["future_5d"])

    # ── 3. 訓練模型 ───────────────────────────
    log("訓練模型...")
    model = RandomForestRegressor(
        n_estimators=300, max_depth=6,
        min_samples_leaf=20, random_state=42, n_jobs=-1
    )
    model.fit(train[FEATURES], train["future_5d"])

    # ── 4. 產生今日訊號 ───────────────────────
    latest_date = panel.dropna(subset=FEATURES).index.max()
    latest = panel.loc[panel.index == latest_date].dropna(subset=FEATURES).copy()
    latest["score"] = model.predict(latest[FEATURES])
    ranking = latest.sort_values("score", ascending=False)

    top5  = ranking.head(5)
    bot5  = ranking.tail(5).sort_values("score")

    # ── 5. 大盤過濾檢查 ───────────────────────
    try:
        twii_raw = yf.download("^TWII", period="3mo",
                    progress=False, auto_adjust=True)
        twii = twii_raw.xs("Close", axis=1, level=0).squeeze()
        twii_ma  = twii.rolling(20).mean()
        mkt_ok   = float(twii.iloc[-1]) > float(twii_ma.iloc[-1])
        mkt_val  = float(twii.iloc[-1])
        mkt_ma   = float(twii_ma.iloc[-1])
        mkt_str  = (f"加權指數 {mkt_val:.0f} > MA20 {mkt_ma:.0f} ✓ 可操作"
                    if mkt_ok else
                    f"加權指數 {mkt_val:.0f} < MA20 {mkt_ma:.0f} ✗ 建議空手")
    except:
        mkt_ok  = True
        mkt_str = "大盤資料取得失敗，預設可操作"

    # ── 6. 今日各股漲跌 ───────────────────────
    try:
        raw2 = yf.download(list(ALL_STOCKS.keys()),
                    period="2d", progress=False, auto_adjust=True)
        raw2 = raw2.xs("Close", axis=1, level=0)
        raw2.columns = [ALL_STOCKS.get(c, c) for c in raw2.columns]
        raw2 = raw2.dropna(axis=1)
        if len(raw2) >= 2:
            today_chg = ((raw2.iloc[-1] - raw2.iloc[-2]) / raw2.iloc[-2] * 100).round(2)
        else:
            today_chg = pd.Series(dtype=float)
    except:
        today_chg = pd.Series(dtype=float)

    # ── 7. 輸出報告 ───────────────────────────
    lines = []
    lines.append(f"{'='*48}")
    lines.append(f"  台股量化每日訊號報告  {today}")
    lines.append(f"{'='*48}")
    lines.append(f"  大盤狀態：{mkt_str}")
    lines.append(f"  資料日期：{latest_date.date()}")
    lines.append(f"  分析股票：{len(names)} 檔")
    lines.append(f"{'='*48}")

    lines.append(f"\n  【模型看多 — 前5名】")
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        name = row["stock"]
        chg  = today_chg.get(name, float("nan"))
        chg_str = f"今日 {chg:+.2f}%" if not pd.isna(chg) else ""
        lines.append(f"  {i}. {name:<12} 分數:{row['score']:+.5f}  {chg_str}")
    lines.append(f"\n  【模型看空 — 後5名】")
    for i, (_, row) in enumerate(bot5.iterrows(), 1):
        name = row["stock"]
        chg  = today_chg.get(name, float("nan"))
        chg_str = f"今日 {chg:+.2f}%" if not pd.isna(chg) else ""
        lines.append(f"  {i}. {name:<12} 分數:{row['score']:+.5f}  {chg_str}")

    lines.append(f"\n{'='*48}")
    report = "\n".join(lines)

    # 印出 + 存檔
    print(report)
    with open(SIGNAL_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    # LINE 推播
    try:
        send_line(report)
        log("LINE 推播完成")
    except Exception as e:
        log(f"LINE 推播失敗：{e}")

    log(f"訊號報告已存檔：{SIGNAL_FILE}")
    log(f"===== 執行完成 =====\n")

if __name__ == "__main__":
    main()