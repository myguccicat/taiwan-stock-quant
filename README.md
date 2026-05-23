# 台股量化分析工具

個人學習量化分析的第一個專案，使用 Python 從零建立。

## 包含功能

- 即時股票掃描（漲跌幅、漲跌停偵測）
- 自動警報系統（波動超過門檻自動提示）
- 歷史股價分析（均線、成交量、累積報酬）
- 均線交叉策略回測（含 Sharpe Ratio、最大回撤）

## 安裝

pip install yfinance pandas matplotlib

## 使用方式

# 每日掃描
python main.py

# 歷史分析
python analysis.py

# 策略回測
python backtest.py

## 回測結果（台積電 2330，近一年）

| 指標 | 數值 |
|------|------|
| 策略累積報酬 | +93.13% |
| 買入持有報酬 | +133.06% |
| Sharpe Ratio | 2.66 |
| 最大回撤 | -12.37% |
| 勝率 | 54.1% |