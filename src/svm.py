"""
Module 3 - Classification: Support Vector Machine
==================================================
⚠️  MUST RUN BEFORE predict.py — saves svm_model.pkl, scaler.pkl
Trains SVM with grid search, saves model files.
Output: models/svm_model.pkl, models/scaler.pkl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
import joblib, os

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
plt.rcParams["figure.dpi"] = 120

print("=" * 55)
print("  MODULE 3 — Support Vector Machine")
print("=" * 55)

# ── Load Data ─────────────────────────────────────────────────────────────────
train = pd.read_csv("data/processed/train.csv")
test  = pd.read_csv("data/processed/test.csv")
le    = joblib.load("models/label_encoder.pkl")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X_train, y_train = train[FEATURES], train["label_enc"]
X_test,  y_test  = test[FEATURES],  test["label_enc"]

# ── Scale Features ────────────────────────────────────────────────────────────
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Binary SVM Demo (rice vs wheat) ──────────────────────────────────────────
print("\n── Binary SVM Demo (rice vs wheat) ─────────────────────")

# Load the original labels to filter by crop name safely
df_raw       = pd.read_csv("data/raw/Crop_recommendation.csv")
train_labels = df_raw.loc[X_train.index, "label"]
test_labels  = df_raw.loc[X_test.index,  "label"]

binary_mask_tr = train_labels.isin(["rice", "wheat"])
binary_mask_te = test_labels.isin(["rice",  "wheat"])

Xb_tr = X_train_sc[binary_mask_tr.values]; yb_tr = y_train[binary_mask_tr]
Xb_te = X_test_sc[binary_mask_te.values];  yb_te = y_test[binary_mask_te]

svm_bin = SVC(kernel="rbf", probability=True)
svm_bin.fit(Xb_tr, yb_tr)
bin_acc = accuracy_score(yb_te, svm_bin.predict(Xb_te))
print(f"   Binary SVM accuracy (rice vs wheat): {bin_acc*100:.2f}%")

# ── Multiclass SVM with Grid Search ──────────────────────────────────────────
print("\n── Grid Search (C, gamma) — this may take 1-2 minutes ──")
param_grid = {
    "C":     [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.001],
}
grid = GridSearchCV(
    SVC(kernel="rbf", probability=True, decision_function_shape="ovr"),
    param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1,
)
grid.fit(X_train_sc, y_train)

print(f"\n✅ Best params : C={grid.best_params_['C']}, gamma={grid.best_params_['gamma']}")
print(f"✅ Best CV acc : {grid.best_score_*100:.2f}%")

# ── Evaluate Best Model ───────────────────────────────────────────────────────
best_svm = grid.best_estimator_
y_pred   = best_svm.predict(X_test_sc)
test_acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
print("\n── Classification Report ────────────────────────────────")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=le.classes_, yticklabels=le.classes_, linewidths=0.3)
plt.title("Confusion Matrix — SVM (RBF Kernel)", fontsize=13, fontweight="bold")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/plots/confusion_matrix_svm.png", bbox_inches="tight")
plt.close()
print("✅ Saved: outputs/plots/confusion_matrix_svm.png")

# ── Save Model Files (needed by predict.py) ───────────────────────────────────
joblib.dump(best_svm, "models/svm_model.pkl")
joblib.dump(scaler,   "models/scaler.pkl")
# label_encoder already saved by 02_feature_engineering.py

print("\n✅ Saved: models/svm_model.pkl")
print("✅ Saved: models/scaler.pkl")
print("✅ Saved: models/label_encoder.pkl  (from step 02)")
print("\n→ You can now run: python src/predict.py")
print("\n✅ SVM Classification complete")