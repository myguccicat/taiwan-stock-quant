# validate_watchlist.py — 驗證所有股票代號

import yfinance as yf
from watchlist import ALL_STOCKS

print(f"共 {len(ALL_STOCKS)} 檔，驗證中（約需 30 秒）...\n")

valid, invalid = {}, {}

for code, name in ALL_STOCKS.items():
    try:
        data = yf.download(code, period="5d", progress=False)
        if len(data) >= 3:
            valid[code] = name
            print(f"  ✓  {name:<12}（{code}）")
        else:
            invalid[code] = name
            print(f"  ✗  {name:<12}（{code}）— 資料不足")
    except Exception as e:
        invalid[code] = name
        print(f"  ✗  {name:<12}（{code}）— 錯誤")

print(f"\n有效：{len(valid)} 檔　無效：{len(invalid)} 檔")

if invalid:
    print("\n需要確認的代號：")
    for code, name in invalid.items():
        print(f"  {name}（{code}）")