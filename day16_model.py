# day16_model.py — Random Forest 預測台股漲跌

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

# ── 1. 載入資料集 ─────────────────────────────
print("載入特徵資料集...")
dataset = pd.read_csv("ml_dataset.csv", index_col=0, parse_dates=True)

FEATURES = [
    "return_1d", "return_5d", "return_20d",
    "ma5_ratio", "ma20_ratio", "ma60_ratio", "ma5_20_cross",
    "vol_5d", "vol_20d", "vol_ratio",
    "rsi14", "bb_position",
    "near_high", "near_low",
]

X = dataset[FEATURES].values
y = dataset["label"].values
dates = dataset.index

print(f"樣本數：{len(X)}，特徵數：{len(FEATURES)}")
print(f"上漲比例：{y.mean()*100:.1f}%")

# ── 2. 時間序列交叉驗證（重要！）────────────────
# 股票資料不能隨機切 train/test，必須用時間順序
# TimeSeriesSplit 確保訓練集永遠在測試集之前
print("\n執行時間序列交叉驗證...")
tscv = TimeSeriesSplit(n_splits=5)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,          # 限制深度，防止過度擬合
    min_samples_leaf=20,  # 每個葉子至少20個樣本
    random_state=42,
    n_jobs=-1
)

cv_scores = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    auc   = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    cv_scores.append(score)
    print(f"  Fold {fold+1}: 準確率={score:.3f}  AUC={auc:.3f}")

print(f"\n平均準確率：{np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

# ── 3. 最終模型：用前 80% 訓練，後 20% 測試 ────
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_test = dates[split:]

model.fit(X_train, y_train)
y_pred  = model.predict(X_test)
y_prob  = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n── 最終測試集結果 ──")
print(f"準確率：{acc:.3f}  AUC：{auc:.3f}")
print(f"\n分類報告：")
print(classification_report(y_test, y_pred,
      target_names=["下跌(0)", "上漲(1)"]))

# ── 4. 特徵重要性 ─────────────────────────────
importances = pd.Series(model.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)

# ── 5. 畫圖 ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Random Forest 台股漲跌預測模型", fontsize=13, fontweight="bold")

# 左上：特徵重要性
ax1 = axes[0, 0]
colors = ["#7C3AED" if i < 5 else "#AFA9EC" for i in range(len(importances))]
ax1.barh(importances.index[::-1], importances.values[::-1],
         color=colors[::-1], alpha=0.85)
ax1.set_title("特徵重要性（模型認為哪些最有用）")
ax1.set_xlabel("重要性分數")
ax1.grid(axis="x", alpha=0.3)

# 右上：混淆矩陣
ax2 = axes[0, 1]
cm = confusion_matrix(y_test, y_pred)
im = ax2.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax2)
for i in range(2):
    for j in range(2):
        ax2.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                 fontsize=14, fontweight="bold",
                 color="white" if cm[i,j] > cm.max()/2 else "black")
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(["預測下跌", "預測上漲"])
ax2.set_yticklabels(["實際下跌", "實際上漲"])
ax2.set_title(f"混淆矩陣（準確率 {acc:.1%}）")

# 左下：ROC 曲線
ax3 = axes[1, 0]
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax3.plot(fpr, tpr, color="#7C3AED", lw=2, label=f"RF模型 AUC={auc:.3f}")
ax3.plot([0,1], [0,1], color="gray", lw=1, linestyle="--", label="隨機猜測 AUC=0.5")
ax3.fill_between(fpr, tpr, alpha=0.1, color="#7C3AED")
ax3.set_xlabel("假正率（False Positive Rate）")
ax3.set_ylabel("真正率（True Positive Rate）")
ax3.set_title("ROC 曲線")
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# 右下：預測機率分佈
ax4 = axes[1, 1]
prob_up   = y_prob[y_test == 1]
prob_down = y_prob[y_test == 0]
ax4.hist(prob_down, bins=25, alpha=0.6, color="#DC2626", label="實際下跌", density=True)
ax4.hist(prob_up,   bins=25, alpha=0.6, color="#16A34A", label="實際上漲", density=True)
ax4.axvline(0.5, color="gray", lw=1, ls="--", label="決策門檻 0.5")
ax4.set_xlabel("模型預測上漲機率")
ax4.set_ylabel("密度")
ax4.set_title("預測機率分佈（分離越好，模型越準）")
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

plt.tight_layout()
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(current_dir, "day16_model.png")
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.show()
print("\n圖表已存檔：day16_model.png")

# ── 6. 儲存模型預測結果 ───────────────────────
result_df = pd.DataFrame({
    "actual":    y_test,
    "predicted": y_pred,
    "prob_up":   y_prob,
}, index=dates_test)
result_df.to_csv("ml_predictions.csv")
print("預測結果已存檔：ml_predictions.csv")