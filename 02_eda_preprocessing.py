"""
PHASE 3–4 — Exploratory Data Analysis & Preprocessing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

df = pd.read_csv("data/soil_biochar_dataset.csv")
Path("plots").mkdir(exist_ok=True)

# ── 1. Basic info ─────────────────────────────────────────────────────────────
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDescriptive Stats:\n", df.describe().round(2))

# ── 2. Distribution of targets by biochar level ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
targets = ["Soil_Health_Index", "Crop_Yield_tha", "Carbon_Sequestered_kgCO2e"]
titles  = ["Soil Health Index", "Crop Yield (t/ha)", "Carbon Sequestered (kg CO₂e/ha)"]

for ax, col, title in zip(axes, targets, titles):
    for b in [0, 2, 5, 10]:
        subset = df[df["Biochar_pct"] == b][col]
        ax.hist(subset, alpha=0.6, bins=20, label=f"{b}% Biochar")
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("plots/target_distributions.png", dpi=150)
plt.close()
print("✅  Plot saved → plots/target_distributions.png")

# ── 3. Correlation heatmap ────────────────────────────────────────────────────
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=150)
plt.close()
print("✅  Plot saved → plots/correlation_heatmap.png")

# ── 4. Boxplot: Soil Health Index vs Biochar % ───────────────────────────────
plt.figure(figsize=(7, 5))
df.boxplot(column="Soil_Health_Index", by="Biochar_pct", grid=False)
plt.title("Soil Health Index by Biochar Treatment")
plt.suptitle("")
plt.xlabel("Biochar (%)")
plt.ylabel("Soil Health Index")
plt.tight_layout()
plt.savefig("plots/shi_vs_biochar.png", dpi=150)
plt.close()
print("✅  Plot saved → plots/shi_vs_biochar.png")

print("\n🎉  EDA complete.")
