"""
PHASE 5 — Explainable AI with SHAP
Generates SHAP summary plots for each target model.
Run AFTER 03_train_models.py
"""

import joblib, shap
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("plots/shap").mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/soil_biochar_dataset.csv")
FEATURES = ["pH", "Nitrogen_kgha", "Phosphorus_kgha", "Potassium_kgha",
            "Moisture_pct", "Organic_Carbon_pct", "EC_dSm", "Biochar_pct"]
X = df[FEATURES]

TARGETS = {
    "Soil_Health_Index":          "Soil Health Index",
    "Crop_Yield_tha":             "Crop Yield (t/ha)",
    "Carbon_Sequestered_kgCO2e":  "Carbon Sequestered",
}

for target_col, target_name in TARGETS.items():
    print(f"\n🔍  SHAP analysis: {target_name}")
    pipe = joblib.load(f"models/{target_col}_best_model.pkl")

    # Transform X through scaler first so SHAP sees model input
    X_scaled = pipe.named_steps["scaler"].transform(X)
    model    = pipe.named_steps["model"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_scaled, feature_names=FEATURES,
                      show=False, plot_type="bar")
    plt.title(f"SHAP Feature Importance — {target_name}")
    plt.tight_layout()
    plt.savefig(f"plots/shap/shap_{target_col}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Beeswarm / dot plot
    plt.figure()
    shap.summary_plot(shap_values, X_scaled, feature_names=FEATURES, show=False)
    plt.title(f"SHAP Summary — {target_name}")
    plt.tight_layout()
    plt.savefig(f"plots/shap/shap_dot_{target_col}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✅  Plots saved → plots/shap/")

print("\n🎉  SHAP analysis complete.")
