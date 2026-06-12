"""
美股早報 LINE Bot 推播腳本
每日由 Windows 工作排程器於台灣時間 8:10 自動執行
"""

import requests
import json
from datetime import datetime

# ── 設定區（只需修改這裡）──────────────────────────────
from config import LINE_CHANNEL_TOKEN, LINE_USER_ID
# ─────────────────────────────────────────────────────


def yf_quote(symbol: str):
    """從 Yahoo Finance 取得最新報價與漲跌幅"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=2d"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=12)
        r.raise_for_status()
        meta = r.json()["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice", 0)
        prev  = meta.get("previousClose") or meta.get("chartPreviousClose", 0)
        pct   = (price - prev) / prev * 100 if prev else 0
        return price, pct
    except Exception:
        return None, None


def fmt_stock(price, pct, show_price=True):
    if price is None:
        return "資料待確認"
    arrow = "▲" if pct >= 0 else "▼"
    if show_price:
        return f"{price:,.2f}（{arrow}{abs(pct):.2f}%）"
    return f"{arrow}{abs(pct):.2f}%"


def fmt_fx(rate):
    return f"{rate:.3f}" if rate else "資料待確認"


def build_message() -> str:
    today = datetime.now().strftime("%Y/%m/%d")

    # 大盤指數
    dji,    dji_pct    = yf_quote("^DJI")
    spx,    spx_pct    = yf_quote("^GSPC")
    ixic,   ixic_pct   = yf_quote("^IXIC")
    sox,    sox_pct    = yf_quote("^SOX")
    vix,    _          = yf_quote("^VIX")

    # 盤前期貨（使用 ES=F / NQ=F / YM=F）
    es,  es_pct  = yf_quote("ES=F")
    nq,  nq_pct  = yf_quote("NQ=F")

    # 匯率（Yahoo Finance 格式：TWDUSD=X 代表 1 USD = ? TWD）
    _, twd_raw = yf_quote("TWD=X")   # Yahoo 給的是 1/rate，需換算
    twd_price, _ = yf_quote("TWD=X")
    jpy_price, _ = yf_quote("JPY=X")
    krw_price, _ = yf_quote("KRW=X")
    cny_price, _ = yf_quote("CNY=X")

    # 石油（WTI）
    oil, oil_pct = yf_quote("CL=F")

    lines = [
        f"📊 美股早報｜{today}",
        "",
        "【昨日收盤】",
        f"• 道瓊    {fmt_stock(dji, dji_pct)}",
        f"• S&P500  {fmt_stock(spx, spx_pct)}",
        f"• 那斯達克 {fmt_stock(ixic, ixic_pct)}",
        f"• 費半SOX  {fmt_stock(sox, sox_pct, show_price=False)}",
        f"• VIX    {f'{vix:.2f}' if vix else '資料待確認'}",
        "",
        "【盤前期貨】",
        f"• S&P500期貨  {fmt_stock(es, es_pct, show_price=False)}",
        f"• Nasdaq期貨  {fmt_stock(nq, nq_pct, show_price=False)}",
        f"• WTI原油     {fmt_stock(oil, oil_pct)}",
        "",
        "【外匯匯率（USD）】",
        f"• 新台幣 TWD  {fmt_fx(twd_price)}",
        f"• 日圓   JPY  {fmt_fx(jpy_price)}",
        f"• 韓元   KRW  {fmt_fx(krw_price)}",
        f"• 人民幣 CNY  {fmt_fx(cny_price)}",
        "",
        "詳細分析請開啟 Cowork 查看完整早報 📱",
    ]
    return "\n".join(lines)


def send_line(message: str) -> bool:
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "to": LINE_USER_ID,
        "messages": [{"type": "text", "text": message}],
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        if r.status_code == 200:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ LINE 推播成功")
            return True
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 失敗 {r.status_code}: {r.text}")
            return False
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 例外: {e}")
        return False


if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 開始取得市場資料...")
    msg = build_message()
    print("─" * 40)
    print(msg)
    print("─" * 40)
    send_line(msg)
