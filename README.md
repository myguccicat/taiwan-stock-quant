# 🇹🇼 台股量化分析系統
> 從零基礎到完整量化交易系統，30 天實戰紀錄

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![ML](https://img.shields.io/badge/ML-RandomForest-green)

---

## 📌 系統概覽

每天早上 8:30 自動執行完整流程：
下載最新股價（47 檔自選股）
↓
訓練 Random Forest 排序模型
↓
產出今日前 5 名選股訊號
↓
存檔 today_signal.txt
↓
LINE Bot 推播到手機

---

## 🗂 專案結構
taiwan-stock-quant/
├── watchlist.py              # 47 檔自選股清單
├── validate_watchlist.py     # 代號驗證工具
├── stock_utils.py            # 股票工具庫
├── main.py                   # 每日掃描器
├── daily_run.py              # 每日自動執行主程式
├── run_daily.bat             # Windows 排程啟動腳本
├── send_market_briefing_line.py  # LINE Bot 推播
├── day24_dashboard.py        # Streamlit 網頁介面
│
├── analysis.py               # 歷史股價分析
├── backtest.py               # 均線策略回測
├── day8_returns.py           # 報酬率分佈分析
├── day9_correlation.py       # 相關性矩陣
├── day10_portfolio.py        # 效率前緣（蒙地卡羅）
├── day11_multi_backtest.py   # 多股票回測
├── day12_stoploss.py         # 停損機制實驗
├── day13_report.py           # 一鍵報告產生器
│
├── day15_features.py         # 特徵工程（14個技術指標）
├── day16_model.py            # RF 分類模型
├── day17_improved_model.py   # 模型優化實驗
├── day18_ranking_model.py    # 多股票排序模型
├── day19_ml_backtest.py      # ML 選股回測系統
│
├── day25_institutional.py    # 三大法人籌碼資料
└── day26_ml_with_institutional.py  # 籌碼面整合實驗

---

## 📊 核心回測結果

### ML 選股策略（最終版本）
- **標的**：47 檔個人自選股
- **測試期間**：2025-08 ～ 2026-06
- **策略**：Random Forest 排序，每 5 日換倉 Top 5

| 指標 | 數值 |
|------|------|
| ML策略年化報酬 | +297.8% |
| 等權基準年化報酬 | +259.1% |
| 超額年化報酬（Alpha） | +38.7% |
| Sharpe Ratio | 2.54 |
| 最大回撤 | -22.7% |
| 勝率 | 56.4% |
| 總交易成本 | 12.66% |

---

## 🔬 實驗紀錄與關鍵發現

### 發現一：效率市場假說驗證
RF 分類模型（預測漲跌方向）AUC 僅約 0.45，低於隨機猜測基準（0.5），
驗證了台股在日線級別的弱式效率市場特性。

### 發現二：排序優於分類
改用回歸模型預測相對強弱排名（Spearman IC +0.0285），
比直接預測漲跌方向更有實際交易價值。

### 發現三：風控在多頭市場反效果
| 風控版本 | 超額報酬 | 結論 |
|---------|---------|------|
| 無風控（最終採用）| +38.7% | ✓ 最佳 |
| 大盤MA20空手過濾 | -14.9% | V型反彈踏空 |
| 平滑倉位風控 | -58.9% | 成本過高 |

### 發現四：籌碼面特徵無顯著效益
加入三大法人買賣超資料後，模型績效幾乎無改變
（+386.1% vs +384.2%），技術面特徵已包含足夠資訊。

---

## 🛠 技術棧

| 類別 | 工具 |
|------|------|
| 語言 | Python 3.11 |
| 資料來源 | yfinance、FinMind |
| 機器學習 | scikit-learn（RandomForest）|
| 資料處理 | pandas、numpy |
| 視覺化 | matplotlib、Streamlit |
| 推播 | LINE Messaging API |
| 自動化 | Windows Task Scheduler |
| 版本控制 | Git / GitHub |

---

## 🚀 快速開始

```bash
# 安裝依賴
pip install yfinance pandas numpy scikit-learn matplotlib streamlit FinMind

# 執行每日訊號
python daily_run.py

# 啟動網頁介面
streamlit run day24_dashboard.py

# 執行回測
python day19_ml_backtest.py
```

---

## 📅 開發歷程（30 天）

| 週次 | 完成內容 |
|------|---------|
| Week 1 | Python 基礎、股票掃描器、第一個回測 |
| Week 2 | pandas 進階、相關性分析、效率前緣、多股票回測 |
| Week 3 | 特徵工程、RF 模型、排序策略、ML 回測系統 |
| Week 4 | 自動排程、LINE 推播、Streamlit 介面、籌碼實驗 |

---

*從零程式基礎出發，30 天完成完整量化交易系統。*