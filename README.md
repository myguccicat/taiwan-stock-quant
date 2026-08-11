# 🇹🇼 台股量化分析系統
> 從零基礎到完整量化交易系統，30 天實戰紀錄 + 持續優化

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![ML](https://img.shields.io/badge/ML-RandomForest-green)

---

## 📌 系統概覽

每天早上 8:30 自動執行完整流程：

```
永豐金 API 同步最新庫存
        ↓
下載最新股價（103 檔 AI 供應鏈股票池）
        ↓
訓練 Random Forest 排序模型（含產業特徵）
        ↓
產出今日前 5 名選股訊號（含產業群組標籤）
        ↓
存檔 today_signal.txt
        ↓
LINE Bot 推播到手機
```

---

## 🗂 專案結構

```
taiwan-stock-quant/
├── watchlist.py              # 103 檔 AI 供應鏈股票清單（自動同步）
├── update_stock_pool.py      # 每週自動篩選更新股票池
├── validate_watchlist.py     # 代號驗證工具
├── stock_utils.py            # 股票工具庫
├── main.py                   # 每日掃描器
├── daily_run.py              # 每日自動執行主程式（含產業特徵）
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
├── day18_ranking_model.py    # 多股票排序模型（5日版本）
├── day19_ml_backtest.py      # ML 選股回測系統
│
├── day25_institutional.py    # 三大法人籌碼資料
├── day26_ml_with_institutional.py  # 籌碼面整合實驗
├── day27_ranking_3day.py     # 排序模型（3日版本）
├── day28_sector_ranking.py   # 排序模型（含產業特徵）★ 主力版本
└── day29_sector_foreign_flow.py   # 外資流向整合實驗
```

---

## 📊 核心回測結果

### ML 選股策略（最終採用版本）
- **標的**：103 檔 AI 供應鏈股票池
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

### 發現五：縮短預測窗口可提升排序訊號強度

| 指標 | 5日週期 | 3日週期 |
|------|--------|--------|
| Spearman Rank IC | +0.0182 | **+0.0649** |
| 多空累積報酬 | -0.2% | +1116.3% |
| 最重要特徵 | near_high（28.8%）| r1（20.2%）|

> ⚠️ 累積報酬未扣交易成本，僅反映模型排序能力。

### 發現六（待確認）：3日換倉+持倉緩衝機制的回測結果存在多變數混雜

同時改動換倉週期（5→3日）、年化計算公式、持倉緩衝機制三個變數，
等權基準報酬從 +259.1% 變為 +183.5%（差距 76 個百分點），
顯示年化公式差異顯著，尚未拆解單一變數的貢獻，數字暫不採用。

- [ ] 固定年化公式，僅切換換倉週期
- [ ] 固定換倉週期，僅切換年化公式
- [ ] 移除緩衝機制，測試純3日換倉

### 發現七：產業群組動能是最強的預測因子

**實驗期間**：2026-07-16 ～ 2026-08-11
**對應檔案**：`day28_sector_ranking.py`、`day29_sector_foreign_flow.py`

將 103 檔股票分為 9 個 AI 供應鏈產業群組，新增三個產業輪動特徵：

| 特徵 | 說明 |
|------|------|
| `sector_momentum` | 群組整體近5日平均報酬 |
| `sector_rank` | 個股在自己群組內的相對排名（0~1）|
| `inter_sector_rank` | 個股超出群組平均的幅度排名 |

**三版本對比：**

| 版本 | Spearman IC | 最重要特徵 | 採用？ |
|------|------------|----------|--------|
| 純技術面（day27）| +0.0206 | r1（22%）| — |
| +產業特徵（day28）| +0.0205 | sector_momentum（**48%**）| ✅ 主力 |
| +外資流向（day29）| +0.0151 | sector_momentum（47%）| ❌ IC下降 |

> **核心發現**：`sector_momentum` 重要性高達 47.58%，遠超過昨日報酬率（r1 僅 19%）。
> **「你所在的產業群組正在漲」比「你昨天漲了多少」更能預測未來3日報酬**——
> 這是台股 AI 供應鏈產業輪動效應的量化驗證。
>
> 外資流向特徵重要性低且加入後 IC 下降，與發現四結論一致，確認不採用。

---

## 🏭 AI 供應鏈產業分群（103 檔股票池）

| 群組 | 代表股票 |
|------|---------|
| 晶片設計 | 聯發科、智原、M31、訊芯-KY |
| 記憶體 | 南亞科、華邦電、旺宏 |
| 製造封測 | 台積電、日月光、穩懋、IET-KY |
| PCB載板 | 南電、欣興、家登 |
| AI伺服器 | 廣達、英業達、緯創、鴻海 |
| 散熱電源 | 奇鋐、貿聯-KY |
| 被動元件 | 國巨、凱美 |
| 半導體設備 | 志聖、商丞、鈦昇 |
| 其他電子 | 技嘉、華碩、群創 |

---

## 🛠 技術棧

| 類別 | 工具 |
|------|------|
| 語言 | Python 3.11 |
| 資料來源 | yfinance、FinMind、永豐金 Shioaji API |
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
pip install yfinance pandas numpy scikit-learn matplotlib streamlit FinMind shioaji

# 每日訊號（含產業特徵）
python daily_run.py

# 排名模型（含產業特徵，主力版本）
python day28_sector_ranking.py

# 每週更新股票池
python update_stock_pool.py

# 啟動網頁介面
streamlit run day24_dashboard.py

# 回測
python day19_ml_backtest.py
```

---

## 📅 開發歷程

| 週次 / 階段 | 完成內容 |
|------------|---------|
| Week 1 | Python 基礎、股票掃描器、第一個回測 |
| Week 2 | pandas 進階、相關性分析、效率前緣、多股票回測 |
| Week 3 | 特徵工程、RF 模型、排序策略、ML 回測系統 |
| Week 4 | 自動排程、LINE 推播、Streamlit 介面、籌碼實驗 |
| 2026-06 | Shioaji API 整合、庫存自動同步、預測窗口敏感度測試 |
| 2026-07 | 股票池從 48 擴展至 103 檔（AI 供應鏈精準篩選）|
| 2026-08 | 產業分群特徵整合，sector_momentum 重要性達 48% |

---

*從零程式基礎出發，持續迭代優化中。*
