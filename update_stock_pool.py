# update_stock_pool.py — 自動篩選 AI 供應鏈股票池
# 建議每週日執行一次，自動更新 watchlist.py 的 MY_WATCHLIST
# 執行：python update_stock_pool.py

import requests
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import date, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

WATCHLIST_PATH = Path(r"C:\Users\user\Desktop\stock\watchlist.py")

# ══════════════════════════════════════════
# 篩選條件設定
# ══════════════════════════════════════════
MIN_PRICE        = 15       # 最低股價（排除雞蛋水餃股）
MAX_PRICE        = 3000     # 最高股價（避免零股資金不足）
MIN_AVG_VOLUME = 5000000  # 5百萬股 ≈ 5000張
MIN_MARKET_CAP   = 50       # 最低市值（億元），排除微型股
# ══════════════════════════════════════════

# AI 供應鏈相關產業代碼（TWSE 產業分類）
TARGET_INDUSTRIES = [
    "23",  # 半導體業
    "24",  # 電子零組件業
    "25",  # 電腦及周邊設備業
    "26",  # 光電業
    "27",  # 通信網路業
    "28",  # 電子通路業
    "29",  # 資訊服務業
    "30",  # 其他電子業
]

# OTC 上櫃代碼清單（需要加 .TWO）
KNOWN_OTC = {
    "3105","3131","3317","3680","4971","6643",
    "6953","7556","8027","8277","3024","6669",
    "3227","6230","3034","6278","3036","4967",
}

def to_yf_code(code: str) -> str:
    return f"{code}.TWO" if code in KNOWN_OTC else f"{code}.TW"

def fetch_twse_stocks() -> pd.DataFrame:
    """從台灣證交所抓上市股票清單"""
    print("📡 抓取上市股票清單（TWSE）...")
    url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        df = pd.DataFrame(data)
        df = df.rename(columns={
            "公司代號": "code",
            "公司名稱": "name",
            "產業別": "industry",
        })
        df["market"] = "TW"
        print(f"   上市：{len(df)} 檔")
        return df[["code","name","industry","market"]]
    except Exception as e:
        print(f"   ❌ 上市清單抓取失敗：{e}")
        return pd.DataFrame()

def fetch_tpex_stocks() -> pd.DataFrame:
    """從櫃買中心抓上櫃股票清單"""
    print("📡 抓取上櫃股票清單（TPEX）...")
    url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis"
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        df = pd.DataFrame(data)
        df = df.rename(columns={
            "SecuritiesCompanyCode": "code",
            "CompanyName": "name",
        })
        df["industry"] = "其他電子業"  # TPEX 產業分類較複雜，先統一標記
        df["market"] = "TWO"
        print(f"   上櫃：{len(df)} 檔")
        return df[["code","name","industry","market"]]
    except Exception as e:
        print(f"   ❌ 上櫃清單抓取失敗：{e}")
        return pd.DataFrame()

def filter_by_industry(df: pd.DataFrame) -> pd.DataFrame:
    """按產業篩選"""
    if df.empty:
        return df
    # 上市：按產業代碼篩選
    twse = df[df["market"] == "TW"]
    tpex = df[df["market"] == "TWO"]

    twse_filtered = twse[twse["industry"].isin(TARGET_INDUSTRIES)]

    # 上櫃：全部保留（之後用成交量過濾）
    result = pd.concat([twse_filtered, tpex], ignore_index=True)
    print(f"   產業篩選後：{len(result)} 檔")
    return result

def filter_by_price_volume(candidates: pd.DataFrame) -> pd.DataFrame:
    """用 yfinance 抓股價和成交量做篩選"""
    print(f"\n📊 下載股價資料篩選（共 {len(candidates)} 檔，需要幾分鐘）...")

    passed = []
    failed = 0

    for i, row in candidates.iterrows():
        code = str(row["code"]).strip()
        yf_code = to_yf_code(code)

        try:
            df = yf.download(yf_code, period="30d",
                             progress=False, auto_adjust=True)
            if df.empty:
                failed += 1
                continue

            # 取得收盤價和成交量
            if hasattr(df.columns, 'levels'):
                close  = df.xs("Close",  axis=1, level=0).iloc[:, 0]
                volume = df.xs("Volume", axis=1, level=0).iloc[:, 0]
            else:
                close  = df["Close"]
                volume = df["Volume"]

            last_price  = float(close.iloc[-1])
            avg_volume = float(volume.rolling(20).mean().iloc[-1])  # 股（不除以1000）

            # 套用篩選條件
            if last_price < MIN_PRICE:
                continue
            if last_price > MAX_PRICE:
                continue
            if avg_volume < MIN_AVG_VOLUME:
                continue

            passed.append({
                "code":      code,
                "yf_code":   yf_code,
                "name":      row["name"],
                "industry":  row["industry"],
                "price":     round(last_price, 1),
                "avg_vol":   round(avg_volume, 0),
            })

            if len(passed) % 20 == 0:
                print(f"   已通過篩選：{len(passed)} 檔...")

        except Exception:
            failed += 1
            continue

        time.sleep(0.1)  # 避免請求太快

    print(f"   篩選完成：通過 {len(passed)} 檔，失敗/排除 {failed} 檔")
    return pd.DataFrame(passed)

def read_existing_watchlist() -> tuple:
    """讀取現有的 MY_HOLDINGS 和 MY_WATCHLIST"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("watchlist", WATCHLIST_PATH)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        holdings  = getattr(mod, "MY_HOLDINGS",  {})
        watchlist = getattr(mod, "MY_WATCHLIST", {})
        return holdings, watchlist
    except Exception as e:
        print(f"❌ 讀取 watchlist.py 失敗：{e}")
        return {}, {}

def update_watchlist_py(new_pool: pd.DataFrame) -> None:
    """更新 watchlist.py 的 MY_WATCHLIST"""
    holdings, old_watchlist = read_existing_watchlist()

    # 建立新的 MY_WATCHLIST
    new_watchlist = {}
    for _, row in new_pool.iterrows():
        yf_code = row["yf_code"]
        name    = row["name"]
        # 不加入已在 MY_HOLDINGS 的股票
        if yf_code not in holdings:
            new_watchlist[yf_code] = name
    
    today = date.today().strftime("%Y-%m-%d")

    # 寫入 watchlist.py
    lines = []
    lines.append(f"# watchlist.py -- auto synced from SinoPac ({today})\n")
    lines.append("# MY_HOLDINGS 自動同步；MY_WATCHLIST 每週自動更新\n\n")

    lines.append("MY_HOLDINGS = {\n")
    for code, name in sorted(holdings.items()):
        otc = "  # OTC" if ".TWO" in code else ""
        lines.append(f'    "{code}":  "{name}",{otc}\n')
    lines.append("}\n\n")

    lines.append("MY_WATCHLIST = {\n")
    for code, name in sorted(new_watchlist.items()):
        otc = "  # OTC" if ".TWO" in code else ""
        lines.append(f'    "{code}":  "{name}",{otc}\n')
    lines.append("}\n\n")

    lines.append("ALL_STOCKS = {**MY_HOLDINGS, **MY_WATCHLIST}\n")

    # 備份
    backup = WATCHLIST_PATH.with_name(f"watchlist_backup_{today}.py")
    backup.write_text(WATCHLIST_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"✅ 備份：{backup.name}")

    WATCHLIST_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"✅ watchlist.py 更新完成")
    print(f"   MY_HOLDINGS：{len(holdings)} 檔")
    print(f"   MY_WATCHLIST：{len(new_watchlist)} 檔")
    print(f"   ALL_STOCKS：{len(holdings) + len(new_watchlist)} 檔")

def main():
    print("=" * 55)
    print("  AI 供應鏈股票池自動更新")
    print(f"  執行日期：{date.today()}")
    print("=" * 55)

    # 1. 抓股票清單
    twse = fetch_twse_stocks()
    tpex = fetch_tpex_stocks()
    all_stocks = pd.concat([twse, tpex], ignore_index=True)

    if all_stocks.empty:
        print("❌ 無法取得股票清單，請檢查網路")
        return

    # 2. 產業篩選
    candidates = filter_by_industry(all_stocks)

    if candidates.empty:
        print("❌ 產業篩選後無候選股票")
        return

    # 3. 價格和成交量篩選（耗時較長）
    passed = filter_by_price_volume(candidates)

    if passed.empty:
        print("❌ 篩選後無股票通過")
        return

    # 4. 顯示結果
    print(f"\n📋 篩選結果摘要：")
    print(passed[["code","name","industry","price","avg_vol"]].to_string(index=False))

    # 5. 更新 watchlist.py
    print(f"\n💾 更新 watchlist.py...")
    update_watchlist_py(passed)

    print("\n" + "=" * 55)
    print("  完成！建議每週日執行一次更新股票池")
    print("=" * 55)

if __name__ == "__main__":
    main()
