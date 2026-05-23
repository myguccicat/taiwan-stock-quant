# main.py — 主程式，負責「使用」工具庫

import pandas as pd
from datetime import date
from stock_utils import scan_stocks, print_report, generate_alerts

WATCHLIST = [
    ("2330.TW", "台積電"),
    ("2317.TW", "鴻海"),
    ("2454.TW", "聯發科"),
    ("2412.TW", "中華電"),
    ("2308.TW", "台達電"),
    ("2882.TW", "國泰金"),
]

# 抓資料
print("正在抓取資料...\n")
results = scan_stocks(WATCHLIST)

# 印報告
print_report(results)

# 今日最強 / 最弱
best = max(results, key=lambda x: x["change_pct"])
worst = min(results, key=lambda x: x["change_pct"])
print(f"\n今日最強：{best['name']} {best['change_pct']:+.2f}%")
print(f"今日最弱：{worst['name']} {worst['change_pct']:+.2f}%")

# 警報系統
alerts = generate_alerts(results, alert_pct=5.0)
print("\n" + "=" * 50)
if alerts:
    print("警報！以下股票波動超過 5%：")
    for a in alerts:
        print(f"  >>> {a['name']} {a['direction']} {a['change_pct']:+.2f}%")
else:
    print("今日無異常波動")
print("=" * 50)

import os  # 如果程式碼最上方沒有 import os，記得補上這行

# === 原本的存成 CSV 段落，請改成這樣 ===
df = pd.DataFrame(results)

# 1. 自動取得目前這個 main.py 檔案所在的資料夾路徑 (即 Desktop\stock)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 把資料夾路徑跟您的檔名組合在一起
filename = os.path.join(current_dir, f"scan_{date.today()}.csv")

# 3. 儲存檔案
df.to_csv(filename, index=False, encoding="utf-8-sig")
print(f"\n結果已儲存：{filename}")