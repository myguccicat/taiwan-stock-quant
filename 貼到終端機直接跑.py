# 貼到終端機直接跑
import yfinance as yf

tickers = [
    "3317.TWO","3680.TWO","6643.TWO","8277.TWO","3131.TWO",
    "8027.TWO","7556.TWO","4971.TWO","3105.TWO","6953.TWO",
]

for t in tickers:
    data = yf.download(t, period="5d", progress=False)
    status = "✓" if len(data) >= 3 else "✗"
    print(f"  {status}  {t}")