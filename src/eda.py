# Exploratory Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Setup ─────────────────────────────────────────────────────────────────────
os.makedirs("outputs/plots", exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

print("=" * 55)
print("  MODULE 1 — Exploratory Data Analysis")
print("=" * 55)

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/Crop_recommendation.csv")

print(f"\n✅ Dataset loaded")
print(f"   Shape      : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Crops      : {df['label'].nunique()} unique classes")
print(f"   Null values: {df.isnull().sum().sum()}")
print(f"   Duplicates : {df.duplicated().sum()}")

print("\n── Feature Summary ──────────────────────────────────────")
print(df.describe().round(2).to_string())

# ── Plot 1: Class Distribution ────────────────────────────────────────────────
print("\n[1/4] Plotting class distribution...")
fig, ax = plt.subplots(figsize=(14, 5))
order = df["label"].value_counts().index
sns.countplot(data=df, y="label", order=order, palette="Greens_r", ax=ax)
ax.set_title("Crop Class Distribution (2200 samples, 22 crops)", fontsize=14, fontweight="bold")
ax.set_xlabel("Count"); ax.set_ylabel("Crop")
plt.tight_layout()
plt.savefig("outputs/plots/eda_class_distribution.png", bbox_inches="tight")
plt.close()
print("   Saved: outputs/plots/eda_class_distribution.png")

# ── Plot 2: Correlation Heatmap ───────────────────────────────────────────────
print("[2/4] Plotting correlation heatmap...")
fig, ax = plt.subplots(figsize=(8, 6))
corr = df.drop("label", axis=1).corr()
mask = [[i < j for j in range(len(corr))] for i in range(len(corr))]
sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGn",
            linewidths=0.5, ax=ax, square=True)
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/plots/eda_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("   Saved: outputs/plots/eda_correlation_heatmap.png")

# ── Plot 3: Feature Boxplots ──────────────────────────────────────────────────
print("[3/4] Plotting feature boxplots...")
features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, feat in enumerate(features):
    sns.boxplot(data=df, x=feat, ax=axes[i], color="#2ECC71")
    axes[i].set_title(feat, fontweight="bold")
axes[-1].set_visible(False)
fig.suptitle("Feature Distributions (Boxplots)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/plots/eda_feature_boxplots.png", bbox_inches="tight")
plt.close()
print("   Saved: outputs/plots/eda_feature_boxplots.png")

# ── Plot 4: Feature Means per Crop ────────────────────────────────────────────
print("[4/4] Plotting feature means per crop...")
crop_means = df.groupby("label")[["N", "P", "K"]].mean().sort_values("N", ascending=False)
crop_means.plot(kind="bar", figsize=(14, 5), colormap="Set2", edgecolor="white")
plt.title("Average N, P, K per Crop", fontsize=13, fontweight="bold")
plt.xlabel("Crop"); plt.ylabel("Average Value (kg/ha)")
plt.xticks(rotation=45, ha="right")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("outputs/plots/eda_npk_per_crop.png", bbox_inches="tight")
plt.close()
print("   Saved: outputs/plots/eda_npk_per_crop.png")

print("\n✅ EDA complete — 4 plots saved to outputs/plots/")
