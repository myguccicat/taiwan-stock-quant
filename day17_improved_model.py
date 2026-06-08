# day17_improved_model.py — 優化版 ML 模型

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

# ── 1. 重新下載更長的資料（3年）─────────────────
print("下載 3 年資料...")
df = yf.download("2330.TW", period="3y", progress=False)
df = df[["Open","High","Low","Close","Volume"]].copy()
df.columns = ["open","high","low","close","volume"]

# ── 2. 特徵工程（加入滯後特徵）──────────────────
def calc_rsi(prices, n=14):
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - (100 / (1 + gain / loss))

# 基礎特徵
df["r1"]  = df["close"].pct_change(1)
df["r5"]  = df["close"].pct_change(5)
df["r20"] = df["close"].pct_change(20)
df["ma5_ratio"]    = df["close"] / df["close"].rolling(5).mean()  - 1
df["ma20_ratio"]   = df["close"] / df["close"].rolling(20).mean() - 1
df["ma60_ratio"]   = df["close"] / df["close"].rolling(60).mean() - 1
df["ma5_20_cross"] = df["close"].rolling(5).mean() / df["close"].rolling(20).mean() - 1
df["vol_5d"]  = df["r1"].rolling(5).std()
df["vol_20d"] = df["r1"].rolling(20).std()
df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
df["rsi14"]  = calc_rsi(df["close"], 14)
bb_mid = df["close"].rolling(20).mean()
bb_std = df["close"].rolling(20).std()
df["bb_position"] = (df["close"] - (bb_mid - 2*bb_std)) / (4*bb_std)
df["near_high"] = df["close"] / df["high"].rolling(20).max()  - 1
df["near_low"]  = df["close"] / df["low"].rolling(20).min()   - 1

# 招式二：滯後特徵（加入昨天、前天的狀態）
BASE = ["r1","ma5_ratio","ma20_ratio","vol_ratio","rsi14","bb_position"]
for col in BASE:
    df[f"{col}_lag1"] = df[col].shift(1)
    df[f"{col}_lag2"] = df[col].shift(2)

# 目標變數
df["future_5d"] = df["close"].pct_change(5).shift(-5)
df["label"] = (df["future_5d"] > 0).astype(int)

# 完整特徵清單
FEATURES = [
    "r1","r5","r20",
    "ma5_ratio","ma20_ratio","ma60_ratio","ma5_20_cross",
    "vol_5d","vol_20d","vol_ratio",
    "rsi14","bb_position","near_high","near_low",
] + [f"{c}_lag{l}" for c in BASE for l in [1,2]]

dataset = df[FEATURES + ["label"]].dropna()
X = dataset[FEATURES].values
y = dataset["label"].values

print(f"樣本數：{len(X)}（比昨天多了 {len(X)-85} 筆）")
print(f"特徵數：{len(FEATURES)}（加入滯後後從 14 → {len(FEATURES)}）")
print(f"上漲比例：{y.mean()*100:.1f}%")

# ── 3. 三種模型比較 ───────────────────────────
models = {
    "RF 原版":      RandomForestClassifier(
                        n_estimators=200, max_depth=5,
                        min_samples_leaf=20, random_state=42),
    # 招式一：class_weight="balanced"
    "RF 平衡權重":  RandomForestClassifier(
                        n_estimators=200, max_depth=5,
                        min_samples_leaf=20, random_state=42,
                        class_weight="balanced"),
    # 加入滯後特徵 + 平衡權重
    "RF 全優化":    RandomForestClassifier(
                        n_estimators=300, max_depth=6,
                        min_samples_leaf=15, random_state=42,
                        class_weight="balanced"),
}

tscv = TimeSeriesSplit(n_splits=5)
print("\n── 交叉驗證比較 ──")
print(f"{'模型':<12} {'平均AUC':>8} {'平均準確率':>10} {'標準差':>8}")
print("─" * 45)

cv_results = {}
for name, clf in models.items():
    aucs, accs = [], []
    for train_idx, test_idx in tscv.split(X):
        clf.fit(X[train_idx], y[train_idx])
        prob = clf.predict_proba(X[test_idx])[:,1]
        pred = clf.predict(X[test_idx])
        aucs.append(roc_auc_score(y[test_idx], prob))
        accs.append(accuracy_score(y[test_idx], pred))
    cv_results[name] = {"auc": aucs, "acc": accs}
    print(f"{name:<12} {np.mean(aucs):>8.3f} {np.mean(accs):>10.3f} {np.std(accs):>8.3f}")

# ── 4. 最佳模型最終測試 ───────────────────────
split = int(len(X) * 0.8)
best_model = models["RF 全優化"]
best_model.fit(X[:split], y[:split])
y_pred = best_model.predict(X[split:])
y_prob = best_model.predict_proba(X[split:])[:,1]
y_test = y[split:]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
print(f"\n── RF 全優化 最終測試結果 ──")
print(f"準確率：{acc:.3f}  AUC：{auc:.3f}")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
precision_down = tn/(tn+fn) if (tn+fn)>0 else 0
recall_down    = tn/(tn+fp) if (tn+fp)>0 else 0
print(f"下跌精確率：{precision_down:.3f}  下跌召回率：{recall_down:.3f}")
print(f"混淆矩陣：\n  TN={tn} FP={fp}\n  FN={fn} TP={tp}")

# 特徵重要性
importances = pd.Series(best_model.feature_importances_, index=FEATURES)
top10 = importances.nlargest(10)

# ── 5. 畫圖 ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("模型優化比較（平衡權重 + 滯後特徵 + 更多資料）",
             fontsize=13, fontweight="bold")

# 左上：AUC 比較（盒鬚圖）
ax1 = axes[0, 0]
auc_data  = [cv_results[n]["auc"] for n in models]
acc_data  = [cv_results[n]["acc"] for n in models]
bp = ax1.boxplot(auc_data, labels=list(models.keys()),
                 patch_artist=True, notch=False)
colors_box = ["#AFA9EC","#7C3AED","#534AB7"]
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.axhline(0.5, color="#EF4444", lw=1, ls="--", label="隨機猜測基準")
ax1.set_title("5-Fold AUC 分佈比較")
ax1.set_ylabel("AUC")
ax1.legend(fontsize=8)
ax1.grid(axis="y", alpha=0.3)

# 右上：ROC 曲線（三模型）
ax2 = axes[0, 1]
split_tmp = int(len(X)*0.8)
line_colors = ["#AFA9EC","#7C3AED","#534AB7"]
for (name, clf), lc in zip(models.items(), line_colors):
    clf.fit(X[:split_tmp], y[:split_tmp])
    prob_tmp = clf.predict_proba(X[split_tmp:])[:,1]
    auc_tmp  = roc_auc_score(y[split_tmp:], prob_tmp)
    fpr, tpr, _ = roc_curve(y[split_tmp:], prob_tmp)
    ax2.plot(fpr, tpr, color=lc, lw=2, label=f"{name} AUC={auc_tmp:.3f}")
ax2.plot([0,1],[0,1], color="gray", lw=1, ls="--", label="隨機猜測")
ax2.set_xlabel("假正率")
ax2.set_ylabel("真正率")
ax2.set_title("三模型 ROC 曲線對比")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# 左下：混淆矩陣（最佳模型）
ax3 = axes[1, 0]
im = ax3.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax3)
labels_cm = [["TN\n(正確預測下跌)", "FP\n(誤判上漲)"],
             ["FN\n(誤判下跌)",     "TP\n(正確預測上漲)"]]
for i in range(2):
    for j in range(2):
        ax3.text(j, i, f"{cm[i,j]}\n{labels_cm[i][j]}",
                 ha="center", va="center", fontsize=9,
                 color="white" if cm[i,j]>cm.max()/2 else "black")
ax3.set_xticks([0,1])
ax3.set_yticks([0,1])
ax3.set_xticklabels(["預測下跌","預測上漲"])
ax3.set_yticklabels(["實際下跌","實際上漲"])
ax3.set_title(f"RF全優化 混淆矩陣（準確率 {acc:.1%}）")

# 右下：Top 10 特徵重要性
ax4 = axes[1, 1]
colors_feat = ["#534AB7" if "_lag" in n else "#7C3AED" for n in top10.index[::-1]]
ax4.barh(top10.index[::-1], top10.values[::-1],
         color=colors_feat, alpha=0.85)
ax4.set_title("Top 10 特徵重要性\n（深色=滯後特徵，淺色=原始特徵）")
ax4.set_xlabel("重要性分數")
ax4.grid(axis="x", alpha=0.3)

plt.tight_layout()
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(current_dir, "day17_improved.png")
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n圖表已存檔：{fname}")