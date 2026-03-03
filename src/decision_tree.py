import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
import joblib, os

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
plt.rcParams["figure.dpi"] = 120

print("=" * 55)
print("  MODULE 3 — Decision Tree (Entropy)")
print("=" * 55)

# ── Load Data ─────────────────────────────────────────────────────────────────
train = pd.read_csv("data/processed/train.csv")
test  = pd.read_csv("data/processed/test.csv")
le    = joblib.load("models/label_encoder.pkl")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X_train, y_train = train[FEATURES], train["label_enc"]
X_test,  y_test  = test[FEATURES],  test["label_enc"]

# ── Tune max_depth (2 to 15) ──────────────────────────────────────────────────
print("\n── Tuning max_depth ────────────────────────────────────")
depths     = range(2, 16)
train_acc  = []
test_acc   = []

for d in depths:
    dt = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, dt.predict(X_train)))
    test_acc.append(accuracy_score(y_test,  dt.predict(X_test)))
    print(f"   depth={d:2d}  Train={train_acc[-1]*100:.1f}%  Test={test_acc[-1]*100:.1f}%")

best_depth = list(depths)[test_acc.index(max(test_acc))]
print(f"\n✅ Best max_depth = {best_depth}  (Test Accuracy = {max(test_acc)*100:.2f}%)")

# ── Plot: Accuracy vs Depth ───────────────────────────────────────────────────
plt.figure(figsize=(9, 4))
plt.plot(list(depths), [a*100 for a in train_acc], "o--", color="#1558A0", label="Train Accuracy")
plt.plot(list(depths), [a*100 for a in test_acc],  "s-",  color="#E74C3C", label="Test Accuracy")
plt.axvline(x=best_depth, color="#F39C12", linestyle=":", linewidth=2, label=f"Best depth={best_depth}")
plt.xlabel("max_depth"); plt.ylabel("Accuracy (%)")
plt.title("Decision Tree — Accuracy vs max_depth", fontweight="bold")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/plots/dt_accuracy_vs_depth.png", bbox_inches="tight")
plt.close()
print("✅ Saved: outputs/plots/dt_accuracy_vs_depth.png")

# ── Train Final Decision Tree ─────────────────────────────────────────────────
best_dt = DecisionTreeClassifier(criterion="entropy", max_depth=best_depth, random_state=42)
best_dt.fit(X_train, y_train)
y_pred  = best_dt.predict(X_test)

# ── Print tree structure (text) ───────────────────────────────────────────────
print("\n── Decision Tree Structure (top 3 levels) ───────────────")
tree_rules = export_text(best_dt, feature_names=FEATURES, max_depth=3)
print(tree_rules)

# ── Feature Importance Plot ───────────────────────────────────────────────────
importances = pd.Series(best_dt.feature_importances_, index=FEATURES).sort_values(ascending=True)
plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="#1558A0", edgecolor="white")
plt.title("Decision Tree — Feature Importances", fontweight="bold")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("outputs/plots/dt_feature_importance.png", bbox_inches="tight")
plt.close()
print("✅ Saved: outputs/plots/dt_feature_importance.png")

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\n── Classification Report ────────────────────────────────")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Save Model ────────────────────────────────────────────────────────────────
joblib.dump(best_dt, "models/decision_tree.pkl")
print("✅ Saved: models/decision_tree.pkl")
print("\n✅ Decision Tree Classification complete")
