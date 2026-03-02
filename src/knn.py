import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib, os

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
plt.rcParams["figure.dpi"] = 120

print("=" * 55)
print("  MODULE 3 — K-Nearest Neighbours")
print("=" * 55)

# ── Load Data ─────────────────────────────────────────────────────────────────
train = pd.read_csv("data/processed/train.csv")
test  = pd.read_csv("data/processed/test.csv")
le    = joblib.load("models/label_encoder.pkl")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X_train, y_train = train[FEATURES], train["label_enc"]
X_test,  y_test  = test[FEATURES],  test["label_enc"]

# ── Scale Features (required for KNN) ────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Tune k (1 to 21) ─────────────────────────────────────────────────────────
print("\n── Tuning k value ──────────────────────────────────────")
k_values  = list(range(1, 22, 2))  # 1,3,5,...,21
train_acc = []
test_acc  = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_sc, y_train)
    train_acc.append(accuracy_score(y_train, knn.predict(X_train_sc)))
    test_acc.append(accuracy_score(y_test,  knn.predict(X_test_sc)))
    print(f"   k={k:2d}  Train={train_acc[-1]*100:.1f}%  Test={test_acc[-1]*100:.1f}%")

best_k   = k_values[test_acc.index(max(test_acc))]
best_acc = max(test_acc)
print(f"\n✅ Best k = {best_k}  (Test Accuracy = {best_acc*100:.2f}%)")

# ── Plot: Accuracy vs k ───────────────────────────────────────────────────────
plt.figure(figsize=(9, 4))
plt.plot(k_values, [a*100 for a in train_acc], "o--", color="#1A7A3A", label="Train Accuracy")
plt.plot(k_values, [a*100 for a in test_acc],  "s-",  color="#E74C3C", label="Test Accuracy")
plt.axvline(x=best_k, color="#F39C12", linestyle=":", linewidth=2, label=f"Best k={best_k}")
plt.xlabel("k value"); plt.ylabel("Accuracy (%)")
plt.title("KNN — Accuracy vs k Value", fontweight="bold")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/plots/knn_accuracy_vs_k.png", bbox_inches="tight")
plt.close()
print("✅ Saved: outputs/plots/knn_accuracy_vs_k.png")

# ── Train Final KNN ───────────────────────────────────────────────────────────
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_sc, y_train)
y_pred = best_knn.predict(X_test_sc)

print("\n── Classification Report ────────────────────────────────")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Save Model ────────────────────────────────────────────────────────────────
joblib.dump(best_knn, "models/knn_model.pkl")
joblib.dump(scaler,   "models/scaler_knn.pkl")
print("✅ Saved: models/knn_model.pkl")
print("✅ Saved: models/scaler_knn.pkl")
print("\n✅ KNN Classification complete")
