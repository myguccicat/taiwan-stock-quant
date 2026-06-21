# day27 紙上模擬交易日誌：隨機森林台股橫斷面選股策略

**紀錄日期**：2026-06-17
**專案目標**：使用機器學習（Random Forest）預測台股未來 3 日報酬率，並執行等權重橫斷面換倉策略。
**當前狀態**：15 萬本金紙上模擬測試期（Paper Trading）。

> 📌 **個人筆記說明**：本文件（`day27_paper_trading_log.md`）為個人交易紙上模擬紀錄，
> 不納入 GitHub 公開 repo。程式碼邏輯與 `day27_ranking_3day.py`（已推送至 GitHub）一致，
> 此處已同步修正排名計算的 bug（原寫死總股數為 12，已改為動態取得實際股票池大小），
> 避免日後對照時產生混淆。
> 數字含義請參照 README 中「發現五」與「發現六」的限制標註，
> 累積報酬未扣交易成本，僅反映模型排序能力，非可實現之真實報酬。

---

## 壹、 策略與參數設定
* **核心演算法**：RandomForestRegressor
* **預測目標**：未來 3 日報酬率 (`future_3d`)
* **換倉週期**：每 3 日換倉一次
* **資金控管**：總預算 150,000 元，嚴格執行等權重 20% 分配（每檔配置約 30,000 元）。
* **防呆機制**：
  1. 導入 `group_by="ticker"` 解決 `yfinance` 欄位錯位問題。
  2. 加入 6~10 名的「智慧緩衝留守區」，避免頻繁洗盤。
  3. 自動匯出每日決策 CSV 檔至 `stock/` 目錄。

---

## 貳、 15萬實戰配置面板 (2026-06-17 收盤結算)

根據 6/17 最新模型打分與真實收盤價，精算出的 15 萬等權重配置名單如下：

| 橫斷面排名 | 股票代碼 / 名稱 | 6/17 收盤價 | 建議委託股數 | 預估投入本金 | 交易類型 | 策略定位與觀察 |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **第 1 名** | **4971 IET-KY** | 526.00 元 | **57 股** | 29,982 元 | 盤中零股 | 三五族群。短線經歷腰斬，現貨拉回止穩，AI 判定 1日動能復甦首選。 |
| **第 2 名** | **3481 群創** | 58.60 元 | **511 股** | 29,945 元 | 盤中零股 | 面板族群。低價低基期防守部位。 |
| **第 3 名** | **2344 華邦電** | 199.00 元 | **150 股** | 29,850 元 | 盤中零股 | 記憶體族群。極短線強勢突破流。 |
| **第 4 名** | **2375 凱美** | 189.50 元 | **158 股** | 29,941 元 | 盤中零股 | 被動元件。高位黏著布林上軌之主升段續抱股。 |
| **第 5 名** | **2408 南亞科** | 437.00 元 | **68 股** | 29,716 元 | 盤中零股 | 記憶體族群。與華邦電組成雙箭頭。 |
| **總計** | | | | **149,434 元** | | *剩餘 566 元作手續費預備金* |

> 🛡️ **智慧緩衝留守區（第 6 ~ 10 名）**
> 訊芯-KY、英業達、聯發科、全新、國巨。
> *(註：若持有上述個股，依紀律不觸發換倉砍股，續抱至跌出 10 名外。)*

---

## 參、 預測核心原始碼 (`day27_ranking_3day.py`)

```python
"""Cross-sectional stock ranking model (Optimized for 15W Real Trading).

This script trains a RandomForestRegressor to rank Taiwan stocks by expected
3-day forward return. It evaluates the model, implements a buffer rank check,
and auto-archives the daily predictions into a desktop folder to prevent column alignment bugs.
"""

from __future__ import annotations

import argparse
import warnings
import os
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

DEFAULT_PERIOD = "3y"
DEFAULT_OUTPUT = "day18_ranking_3day.png"
DEFAULT_CACHE_DIR = ".yfinance_cache"
FORWARD_DAYS = 3
TEST_SIZE = 0.2
TOP_N = 5
RANDOM_STATE = 42

from watchlist import ALL_STOCKS
STOCKS = ALL_STOCKS

FEATURE_COLUMNS = [
    "r1", "r5", "r20", "ma5_ratio", "ma20_ratio", "ma60_ratio",
    "vol_ratio", "vol_5d", "rsi14", "bb_pos", "near_high",
]

@dataclass(frozen=True)
class