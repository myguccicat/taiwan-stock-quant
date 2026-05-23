# stock_utils.py — 你的第一個量化工具庫

import yfinance as yf

def check_limit(yesterday, today):
    """判斷股票漲跌狀態"""
    change_pct = (today - yesterday) / yesterday * 100

    if change_pct >= 10:
        status = "漲停"
    elif change_pct <= -10:
        status = "跌停"
    elif change_pct > 0:
        status = "上漲"
    elif change_pct < 0:
        status = "下跌"
    else:
        status = "平盤"

    return status, round(change_pct, 2)


def get_stock_info(code, name):
    """抓取單一股票的昨收和今收"""
    data = yf.download(code, period="2d", progress=False)

    if len(data) < 2:
        return None

    yesterday = float(data["Close"].iloc[-2].item())
    today = float(data["Close"].iloc[-1].item())
    status, pct = check_limit(yesterday, today)

    return {
        "name": name,
        "code": code,
        "yesterday": round(yesterday, 1),
        "today": round(today, 1),
        "status": status,
        "change_pct": pct,
    }


def scan_stocks(stock_list):
    """掃描多檔股票，回傳結果清單"""
    results = []
    for code, name in stock_list:
        info = get_stock_info(code, name)
        if info:
            results.append(info)
    return results


def print_report(results):
    """印出整齊的報告"""
    print("=" * 50)
    print(f"{'股票':<6} {'狀態':<6} {'漲跌幅':>8} {'今日收盤':>10}")
    print("=" * 50)
    for r in results:
        print(f"{r['name']:<6} {r['status']:<6} {r['change_pct']:>+7.2f}% {r['today']:>10.1f}")
    print("=" * 50)

def generate_alerts(results, alert_pct=5.0):
    """產生警報：漲跌超過門檻就觸發"""
    alerts = []
    for r in results:
        if abs(r["change_pct"]) >= alert_pct:
            direction = "大漲" if r["change_pct"] > 0 else "大跌"
            alerts.append({
                "name": r["name"],
                "direction": direction,
                "change_pct": r["change_pct"],
            })
    return alerts