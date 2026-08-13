# update_stock_pool.py — 自動篩選 AI 供應鏈股票池（依偏好優化版）
# 建議每週日執行一次，自動更新 watchlist.py 的 MY_WATCHLIST
# 執行：python update_stock_pool.py

import requests
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import date
import time
import warnings
warnings.filterwarnings("ignore")

WATCHLIST_PATH = Path(r"C:\Users\user\Desktop\stock\watchlist.py")

# ══════════════════════════════════════════
# 篩選條件設定
# ══════════════════════════════════════════
MIN_PRICE      = 15          # 最低股價
MAX_PRICE      = 5000        # 最高股價
MIN_AVG_VOLUME = 5_000_000   # 近20日平均成交量（股），約5000張

# 黑名單：明確排除非 AI 供應鏈股票
BLACKLIST = {
    "3532",  # 台塑勝高科技
    "6505",  # 台塑石化
    "9911",  # 台灣汽電共生
    "2305",  # 全友電腦
    "2332",  # 友訊科技
}
# ══════════════════════════════════════════

# 上市股票：只保留 AI 硬體供應鏈核心產業
TWSE_TARGET = {
    "23",  # 半導體業
    "24",  # 電子零組件業
    "25",  # 電腦及周邊設備業
    "30",  # 其他電子業
}

# 你的偏好自選股（來自 6/30 備份，全部保留）
PREFERRED_WATCHLIST = {
    # AI 伺服器供應鏈
    "2382.TW":  "廣達",
    "2356.TW":  "英業達",
    "3231.TW":  "緯創",
    "2317.TW":  "鴻海",
    "6414.TW":  "樺漢",
    "2395.TW":  "研華",

    # 半導體設計
    "2454.TW":  "聯發科",
    "3661.TW":  "世芯-KY",
    "6451.TW":  "訊芯-KY",
    "6533.TW":  "晶心科",
    "2301.TW":  "光寶科",
    "2359.TW":  "所羅門",

    # 記憶體
    "2337.TW":  "旺宏",
    "2344.TW":  "華邦電",
    "2408.TW":  "南亞科",

    # 被動元件
    "2327.TW":  "國巨",
    "2375.TW":  "凱美",

    # PCB / 載板
    "8046.TW":  "南電",
    "3037.TW":  "欣興",
    "4958.TW":  "臻鼎-KY",

    # 散熱 / 連接器
    "3017.TW":  "奇鋐",
    "3665.TW":  "貿聯-KY",

    # 半導體製造 / 封測
    "2330.TW":  "台積電",
    "3711.TW":  "日月光投控",
    "6239.TW":  "力成",
    "3035.TW":  "智原",

    # 半導體設備 / 材料
    "2467.TW":  "志聖",
    "1560.TW":  "中砂",
    "2376.TW":  "技嘉",
    "2357.TW":  "華碩",
    "3013.TW":  "晟銘電",

    # 其他（你有持倉或特別關注）
    "1319.TW":  "東陽",
    "1568.TW":  "倉佑",
    "2637.TW":  "慧洋-KY",
    "2498.TW":  "宏達電",
    "2455.TW":  "全新",

    # OTC 上櫃 AI 供應鏈
    "6643.TWO": "M31",
    "8277.TWO": "商丞",
    "3131.TWO": "弘塑",
    "8027.TWO": "鈦昇",
    "7556.TWO": "意德士",
    "3680.TWO": "家登",
    "3105.TWO": "穩懋",
    "6953.TWO": "家碩",
    "4971.TWO": "IET-KY",
    "3317.TWO": "尼克森",

    # OTC 其他 AI 相關
    "6230.TW":  "超豐",
    "3034.TW":  "聯詠",
    "3036.TW":  "文曄",
    "6278.TW":  "台表科",
    "3227.TWO": "原相",
    "6669.TW":  "緯穎",
}


def to_yf_code(code: str) -> str:
    return code if (".TW" in code or ".TWO" in code) else f"{code}.TW"


def fetch_twse_stocks() -> pd.DataFrame:
    """從台灣證交所抓上市股票清單"""
    print("📡 抓取上市股票清單（TWSE）...")
    url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    try:
        r = requests.get(url, timeout=15)
        df = pd.DataFrame(r.json())
        df = df.rename(columns={
            "公司代號": "code",
            "公司名稱": "name",
            "產業別":  "industry",
        })
        df = df[df["industry"].isin(TWSE_TARGET)].copy()
        df["is_preferred"] = False
        print(f"   上市（產業篩選後）：{len(df)} 檔")
        return df[["code", "name", "industry", "is_preferred"]]
    except Exception as e:
        print(f"   ❌ 上市清單抓取失敗：{e}")
        return pd.DataFrame()


def get_preferred_candidates() -> pd.DataFrame:
    """你的偏好自選股，直接加入不受成交量限制"""
    print("📡 載入偏好自選股清單...")
    rows = []
    for code, name in PREFERRED_WATCHLIST.items():
        pure_code = code.replace(".TW","").replace(".TWO","")
        rows.append({
            "code":         pure_code,
            "name":         name,
            "industry":     "preferred",
            "is_preferred": True,
            "yf_code":      code,
        })
    df = pd.DataFrame(rows)
    print(f"   偏好自選股：{len(df)} 檔（不受成交量限制）")
    return df


def filter_by_price_volume(candidates: pd.DataFrame) -> pd.DataFrame:
    """用 yfinance 抓股價和成交量做篩選"""
    preferred = candidates[candidates["is_preferred"] == True].copy()
    normal    = candidates[candidates["is_preferred"] == False].copy()

    print(f"\n📊 下載股價資料篩選...")
    print(f"   偏好自選股：{len(preferred)} 檔（只驗證股價）")
    print(f"   市場篩選股：{len(normal)} 檔（需通過成交量門檻）")

    passed = []
    failed = 0

    all_candidates = pd.concat([preferred, normal], ignore_index=True)

    for _, row in all_candidates.iterrows():
        code         = str(row["code"]).strip()
        is_preferred = bool(row["is_preferred"])

        # 黑名單過濾
        if code in BLACKLIST:
            print(f"   🚫 黑名單排除：{code} {row['name']}")
            continue

        if "yf_code" in row and pd.notna(row.get("yf_code")):
            yf_code = row["yf_code"]
        else:
            yf_code = f"{code}.TW"

        try:
            df = yf.download(yf_code, period="30d",
                             progress=False, auto_adjust=True)
            if df.empty:
                failed += 1
                continue

            if hasattr(df.columns, "levels"):
                close  = df.xs("Close",  axis=1, level=0).iloc[:, 0]
                volume = df.xs("Volume", axis=1, level=0).iloc[:, 0]
            else:
                close  = df["Close"]
                volume = df["Volume"]

            last_price = float(close.iloc[-1])
            avg_vol    = float(volume.rolling(20).mean().iloc[-1])

            if last_price < MIN_PRICE or last_price > MAX_PRICE:
                if not is_preferred:
                    continue
                print(f"   ⚠️ {code} 股價 {last_price:.1f} 超出範圍，但仍保留（偏好股）")

            if not is_preferred and avg_vol < MIN_AVG_VOLUME:
                continue

            passed.append({
                "code":         code,
                "yf_code":      yf_code,
                "name":         row["name"],
                "industry":     row["industry"],
                "price":        round(last_price, 1),
                "avg_vol_M":    round(avg_vol / 1_000_000, 2),
                "is_preferred": is_preferred,
            })

            if len(passed) % 20 == 0:
                print(f"   已通過：{len(passed)} 檔...")

        except Exception:
            failed += 1
            continue

        time.sleep(0.05)

    print(f"   篩選完成：通過 {len(passed)} 檔，失敗/排除 {failed} 檔")
    return pd.DataFrame(passed)


def read_existing_holdings() -> dict:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("watchlist", WATCHLIST_PATH)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "MY_HOLDINGS", {})
    except Exception as e:
        print(f"❌ 讀取 watchlist.py 失敗：{e}")
        return {}


def update_watchlist_py(new_pool: pd.DataFrame) -> None:
    holdings = read_existing_holdings()

    new_watchlist = {}
    for _, row in new_pool.iterrows():
        yf_code = row["yf_code"]
        name    = row["name"]
        if yf_code not in holdings:
            new_watchlist[yf_code] = name

    today = date.today().strftime("%Y-%m-%d")

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

    backup = WATCHLIST_PATH.with_name(f"watchlist_backup_{today}.py")
    backup.write_text(WATCHLIST_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"✅ 備份：{backup.name}")

    WATCHLIST_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"✅ watchlist.py 更新完成")
    print(f"   MY_HOLDINGS ：{len(holdings)} 檔")
    print(f"   MY_WATCHLIST：{len(new_watchlist)} 檔")
    print(f"   ALL_STOCKS  ：{len(holdings) + len(new_watchlist)} 檔")

    preferred_in = new_pool[new_pool["is_preferred"] == True]
    print(f"\n✅ 偏好自選股確認（{len(preferred_in)} 檔全數保留）")


def main():
    print("=" * 55)
    print("  AI 供應鏈股票池自動更新（偏好優化版）")
    print(f"  執行日期：{date.today()}")
    print("=" * 55)

    preferred  = get_preferred_candidates()
    twse       = fetch_twse_stocks()

    preferred_codes = set(preferred["code"].tolist())
    twse_extra = twse[~twse["code"].isin(preferred_codes)].copy()
    twse_extra["yf_code"] = twse_extra["code"].apply(lambda c: f"{c}.TW")

    candidates = pd.concat([preferred, twse_extra], ignore_index=True)
    print(f"\n   候選總數：{len(candidates)} 檔（偏好 {len(preferred)} + 市場補充 {len(twse_extra)}）")

    passed = filter_by_price_volume(candidates)

    if passed.empty:
        print("❌ 篩選後無股票通過")
        return

    print(f"\n💾 更新 watchlist.py...")
    update_watchlist_py(passed)

    print("\n" + "=" * 55)
    print("  完成！建議每週日執行一次")
    print("=" * 55)


if __name__ == "__main__":
    main()