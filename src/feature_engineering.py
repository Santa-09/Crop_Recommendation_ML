import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 55)
print("  MODULE 1/2 — Feature Engineering")
print("=" * 55)

# ── Load Raw Data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/Crop_recommendation.csv")
print(f"\n✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ── Feature 1: NPK Ratio ──────────────────────────────────────────────────────
df["NPK_ratio"] = df["N"] / (df["P"] + df["K"] + 1)
print("✅ Feature 1 created: NPK_ratio")

# ── Feature 2: Temperature-Humidity Index (THI) ───────────────────────────────
df["THI"] = df["temperature"] * df["humidity"] / 100
print("✅ Feature 2 created: THI (Temp-Humidity Index)")

# ── Feature 3: pH Category ────────────────────────────────────────────────────
df["pH_cat"] = pd.cut(df["ph"],
                       bins=[0, 6.0, 7.5, 14],
                       labels=["Acidic", "Neutral", "Alkaline"])
pH_dummies = pd.get_dummies(df["pH_cat"], prefix="pH")
df = pd.concat([df, pH_dummies], axis=1).drop("pH_cat", axis=1)
print("✅ Feature 3 created: pH_cat → Acidic / Neutral / Alkaline (one-hot)")

# ── Feature 4: Rainfall Zone ─────────────────────────────────────────────────
df["rainfall_zone"] = pd.cut(df["rainfall"],
                               bins=[0, 80, 150, 9999],
                               labels=["Low", "Medium", "High"])
rain_dummies = pd.get_dummies(df["rainfall_zone"], prefix="rain")
df = pd.concat([df, rain_dummies], axis=1).drop("rainfall_zone", axis=1)
print("✅ Feature 4 created: rainfall_zone → Low / Medium / High (one-hot)")

# ── Correlation Pruning ───────────────────────────────────────────────────────
num_cols = df.select_dtypes(include=[np.number]).drop("label" if "label" in df.columns else [], axis=1, errors="ignore")
corr_matrix = num_cols.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.90)]
if to_drop:
    df.drop(columns=to_drop, inplace=True)
    print(f"✅ Dropped highly correlated features: {to_drop}")
else:
    print("✅ No features dropped (no correlation > 0.90)")

# ── Derive Yield Score (Regression Target) ────────────────────────────────────
df["yield_score"] = (
    (df["N"] - df["N"].min()) / (df["N"].max() - df["N"].min()) +
    (df["P"] - df["P"].min()) / (df["P"].max() - df["P"].min()) +
    (df["K"] - df["K"].min()) / (df["K"].max() - df["K"].min()) +
    (df["rainfall"] - df["rainfall"].min()) / (df["rainfall"].max() - df["rainfall"].min())
) / 4
print("✅ yield_score derived (normalized sum of N+P+K+rainfall)")

# ── Encode Crop Labels ────────────────────────────────────────────────────────
le = LabelEncoder()
df["label_enc"] = le.fit_transform(df["label"])
joblib.dump(le, "models/label_encoder.pkl")
print(f"✅ Labels encoded: {len(le.classes_)} classes")
print(f"   Saved: models/label_encoder.pkl")

# ── Save Engineered Dataset ───────────────────────────────────────────────────
df.to_csv("data/processed/features_engineered.csv", index=False)
print(f"✅ Saved: data/processed/features_engineered.csv  ({df.shape})")

# ── Train / Test Split ────────────────────────────────────────────────────────
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall",
            "NPK_ratio", "THI"]
# Add any one-hot columns that exist
extra_cols = [c for c in df.columns if c.startswith("pH_") or c.startswith("rain_")]
FEATURES += extra_cols

X = df[FEATURES]
y = df["label_enc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = X_train.copy(); train_df["label_enc"] = y_train.values
test_df  = X_test.copy();  test_df["label_enc"]  = y_test.values
train_df["yield_score"] = df.loc[X_train.index, "yield_score"].values
test_df["yield_score"]  = df.loc[X_test.index,  "yield_score"].values

train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv",  index=False)

print(f"✅ Train split: {len(X_train)} rows → data/processed/train.csv")
print(f"✅ Test split : {len(X_test)} rows  → data/processed/test.csv")
print(f"\n✅ Feature Engineering complete — features used: {FEATURES}")
