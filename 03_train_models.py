"""
PHASE 4 — AI Model Training & Evaluation
Trains three models for each target variable:
  - Random Forest
  - Gradient Boosting
  - XGBoost

Outputs: model performance table + feature importance plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, joblib
from pathlib import Path

from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── Setup ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/soil_biochar_dataset.csv")
Path("models").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)

FEATURES = ["pH", "Nitrogen_kgha", "Phosphorus_kgha", "Potassium_kgha",
            "Moisture_pct", "Organic_Carbon_pct", "EC_dSm", "Biochar_pct"]

TARGETS = {
    "Soil_Health_Index":          "Soil Health Index",
    "Crop_Yield_tha":             "Crop Yield (t/ha)",
    "Carbon_Sequestered_kgCO2e":  "Carbon Sequestered (kg CO₂e/ha)",
}

X = df[FEATURES]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

MODELS = {
    "Random Forest":       RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting":   GradientBoostingRegressor(n_estimators=200, random_state=42),
    "XGBoost":             XGBRegressor(n_estimators=200, random_state=42,
                                        verbosity=0, eval_metric="rmse"),
}

results = []
best_models = {}  # target → best model pipeline

# ── Training loop ─────────────────────────────────────────────────────────────
for target_col, target_name in TARGETS.items():
    y = df[target_col]
    print(f"\n{'='*55}")
    print(f"  Target: {target_name}")
    print(f"{'='*55}")

    best_r2, best_name, best_pipe = -np.inf, None, None

    for model_name, model in MODELS.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])
        cv_results = cross_validate(
            pipe, X, y, cv=kf,
            scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
            return_train_score=False, n_jobs=-1
        )
        r2   =  cv_results["test_r2"].mean()
        rmse = -cv_results["test_neg_root_mean_squared_error"].mean()
        mae  = -cv_results["test_neg_mean_absolute_error"].mean()

        print(f"  {model_name:<22} R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
        results.append({"Target": target_name, "Model": model_name,
                        "R2": round(r2,4), "RMSE": round(rmse,4), "MAE": round(mae,4)})

        if r2 > best_r2:
            best_r2, best_name, best_pipe = r2, model_name, pipe

    # Fit best model on full data and save
    best_pipe.fit(X, y)
    best_models[target_col] = best_pipe
    safe_col = target_col.replace(" ", "_")
    joblib.dump(best_pipe, f"models/{safe_col}_best_model.pkl")
    print(f"\n  ✅  Best: {best_name} (R²={best_r2:.4f}) — saved to models/")

    # ── Feature importance plot ───────────────────────────────────────────────
    inner_model = best_pipe.named_steps["model"]
    if hasattr(inner_model, "feature_importances_"):
        fi = pd.Series(inner_model.feature_importances_, index=FEATURES).sort_values()
        plt.figure(figsize=(7, 4))
        fi.plot(kind="barh", color="steelblue")
        plt.title(f"Feature Importance — {target_name}\n({best_name})")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(f"plots/fi_{safe_col}.png", dpi=150)
        plt.close()

# ── Results table ─────────────────────────────────────────────────────────────
res_df = pd.DataFrame(results)
res_df.to_csv("models/model_comparison.csv", index=False)
print("\n\n📊 Full Model Comparison:")
print(res_df.to_string(index=False))
print("\n✅  Saved → models/model_comparison.csv")
print("✅  Feature importance plots → plots/")
