import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
plt.rcParams["figure.dpi"] = 120

print("=" * 55)
print("  MODULE 2 — Regression (Yield Prediction)")
print("=" * 55)

# ── Load Data ─────────────────────────────────────────────────────────────────
train = pd.read_csv("data/processed/train.csv")
test  = pd.read_csv("data/processed/test.csv")

FEATURES_ORIG = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FEATURES_ENG  = FEATURES_ORIG + ["NPK_ratio", "THI"]
TARGET = "yield_score"

X_train_orig = train[FEATURES_ORIG];  y_train = train[TARGET]
X_test_orig  = test[FEATURES_ORIG];   y_test  = test[TARGET]
X_train_eng  = train[FEATURES_ENG]
X_test_eng   = test[FEATURES_ENG]

def evaluate(name, y_true, y_pred):
    return {
        "Model": name,
        "RMSE":  round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAE":   round(mean_absolute_error(y_true, y_pred), 4),
        "R²":    round(r2_score(y_true, y_pred), 4),
    }

results = []

# ── Model 1: Simple Linear Regression (Rainfall only) ────────────────────────
print("\n[1/4] Simple Linear Regression (Rainfall → Yield)...")
slr = LinearRegression()
slr.fit(train[["rainfall"]], y_train)
y_pred_slr = slr.predict(test[["rainfall"]])
results.append(evaluate("Simple LR (Rainfall only)", y_test, y_pred_slr))
print(f"   R² = {results[-1]['R²']}")

# ── Model 2: Multiple Linear Regression ──────────────────────────────────────
print("[2/4] Multiple Linear Regression (7 original features)...")
mlr = LinearRegression()
mlr.fit(X_train_orig, y_train)
y_pred_mlr = mlr.predict(X_test_orig)
results.append(evaluate("Multiple LR (7 features)", y_test, y_pred_mlr))
print(f"   R² = {results[-1]['R²']}")

# ── Model 3: MLR with Engineered Features ────────────────────────────────────
print("[3/4] Multiple LR with Engineered Features (9 features)...")
mlr_eng = LinearRegression()
mlr_eng.fit(X_train_eng, y_train)
y_pred_mlr_eng = mlr_eng.predict(X_test_eng)
results.append(evaluate("Multiple LR + Engineered", y_test, y_pred_mlr_eng))
print(f"   R² = {results[-1]['R²']}")

# ── Model 4: Polynomial Regression (Non-Linear) ───────────────────────────────
print("[4/4] Polynomial Regression (degree=2, original features)...")
poly_pipe = Pipeline([
    ("poly",  PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler",StandardScaler()),
    ("lr",    LinearRegression()),
])
poly_pipe.fit(X_train_orig, y_train)
y_pred_poly = poly_pipe.predict(X_test_orig)
results.append(evaluate("Polynomial LR (deg=2)", y_test, y_pred_poly))
print(f"   R² = {results[-1]['R²']}")

# ── PCA Comparison ────────────────────────────────────────────────────────────
print("\n── PCA Analysis ─────────────────────────────────────────")
scaler_pca = StandardScaler()
X_train_sc  = scaler_pca.fit_transform(X_train_eng)
X_test_sc   = scaler_pca.transform(X_test_eng)

for n in [3, 5, 7]:
    pca = PCA(n_components=n)
    X_tr_pca = pca.fit_transform(X_train_sc)
    X_te_pca = pca.transform(X_test_sc)
    lr_pca   = LinearRegression().fit(X_tr_pca, y_train)
    y_p      = lr_pca.predict(X_te_pca)
    r2       = round(r2_score(y_test, y_p), 4)
    var      = round(pca.explained_variance_ratio_.sum() * 100, 1)
    print(f"   PCA n={n}: R²={r2}, Variance explained={var}%")
    results.append(evaluate(f"MLR + PCA (n={n})", y_test, y_p))

# ── Save Results ──────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/reports/regression_metrics.csv", index=False)
print(f"\n✅ Saved: outputs/reports/regression_metrics.csv")
print(results_df.to_string(index=False))

# ── Plot: Actual vs Predicted ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, preds, title in zip(axes,
    [y_pred_mlr, y_pred_mlr_eng],
    ["Multiple LR (7 features)", "Multiple LR + Engineered"]):
    ax.scatter(y_test, preds, alpha=0.4, color="#1A7A3A", s=15)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1)
    ax.set_xlabel("Actual Yield Score"); ax.set_ylabel("Predicted")
    ax.set_title(title, fontweight="bold")
plt.suptitle("Actual vs Predicted — Regression Models", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/plots/regression_actual_vs_pred.png", bbox_inches="tight")
plt.close()
print("✅ Saved: outputs/plots/regression_actual_vs_pred.png")

# ── Plot: Metrics Bar Chart ───────────────────────────────────────────────────
top4 = results_df[results_df["Model"].str.contains("LR|Poly")].head(4)
fig, ax = plt.subplots(figsize=(10, 4))
x = range(len(top4))
ax.bar([i - 0.2 for i in x], top4["RMSE"], width=0.2, label="RMSE", color="#E74C3C")
ax.bar([i       for i in x], top4["MAE"],  width=0.2, label="MAE",  color="#F39C12")
ax.bar([i + 0.2 for i in x], top4["R²"],   width=0.2, label="R²",   color="#27AE60")
ax.set_xticks(list(x)); ax.set_xticklabels(top4["Model"], rotation=15, ha="right", fontsize=9)
ax.legend(); ax.set_title("Regression Metrics Comparison", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/plots/regression_metrics_comparison.png", bbox_inches="tight")
plt.close()
print("✅ Saved: outputs/plots/regression_metrics_comparison.png")
print("\n✅ Module 2 — Regression complete")
