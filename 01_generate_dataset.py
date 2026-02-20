"""
PHASE 3 — Synthetic Dataset Generation
Project: AI-Based Soil Optimization Using Biochar from Prosopis juliflora
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 500  # number of samples

# ── Biochar treatment groups ──────────────────────────────────────────────────
biochar_pct = np.random.choice([0, 2, 5, 10], size=N, p=[0.25, 0.25, 0.25, 0.25])

# ── Base soil parameters (biochar improves values progressively) ──────────────
improvement = biochar_pct / 10  # scale 0→0, 2→0.2, 5→0.5, 10→1.0

pH          = np.clip(np.random.normal(6.5, 0.6, N) + improvement * 0.4, 5.0, 8.5)
nitrogen    = np.clip(np.random.normal(180, 40, N)  + improvement * 60,  80, 400)   # kg/ha
phosphorus  = np.clip(np.random.normal(20,  8,  N)  + improvement * 15,  5,  80)    # kg/ha
potassium   = np.clip(np.random.normal(140, 30, N)  + improvement * 40,  60, 320)   # kg/ha
moisture    = np.clip(np.random.normal(30,  8,  N)  + improvement * 12,  10, 65)    # %
org_carbon  = np.clip(np.random.normal(0.6, 0.2, N) + improvement * 0.5, 0.2, 2.5) # %
ec          = np.clip(np.random.normal(0.5, 0.15,N) - improvement * 0.05,0.1, 1.2) # dS/m

# ── Target 1: Soil Health Index (0–100) ───────────────────────────────────────
# Weighted combination of normalised parameters
pH_score  = 100 - np.abs(pH - 6.8) * 20          # optimal ≈ 6.8
n_score   = np.clip(nitrogen  / 400 * 100, 0, 100)
p_score   = np.clip(phosphorus / 80  * 100, 0, 100)
k_score   = np.clip(potassium  / 320 * 100, 0, 100)
m_score   = np.clip(moisture   / 65  * 100, 0, 100)
oc_score  = np.clip(org_carbon / 2.5 * 100, 0, 100)
ec_score  = 100 - np.clip(ec / 1.2 * 100, 0, 100)

soil_health_index = (
    0.20 * pH_score +
    0.20 * n_score  +
    0.15 * p_score  +
    0.15 * k_score  +
    0.15 * m_score  +
    0.10 * oc_score +
    0.05 * ec_score
)
soil_health_index = np.clip(soil_health_index + np.random.normal(0, 3, N), 0, 100)

# ── Target 2: Estimated Crop Yield (tonnes/ha) ────────────────────────────────
crop_yield = np.clip(
    1.5 + soil_health_index * 0.04 + improvement * 0.8 + np.random.normal(0, 0.3, N),
    0.5, 7.0
)

# ── Target 3: Carbon Sequestered (kg CO₂e / ha) ───────────────────────────────
# Formula: biochar_applied × carbon_fraction × stability_factor × (44/12)
biochar_applied_kg = biochar_pct * 100          # assume 100 kg per 1% per ha
carbon_fraction    = 0.75                        # juliflora biochar ~75% carbon
stability_factor   = 0.85                        # fraction stable over 100 yrs
carbon_sequestered = (
    biochar_applied_kg * carbon_fraction * stability_factor * (44/12)
    + np.random.normal(0, 50, N)
)
carbon_sequestered = np.clip(carbon_sequestered, 0, None)

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    "pH":                   pH.round(2),
    "Nitrogen_kgha":        nitrogen.round(1),
    "Phosphorus_kgha":      phosphorus.round(1),
    "Potassium_kgha":       potassium.round(1),
    "Moisture_pct":         moisture.round(1),
    "Organic_Carbon_pct":   org_carbon.round(3),
    "EC_dSm":               ec.round(3),
    "Biochar_pct":          biochar_pct,
    "Soil_Health_Index":    soil_health_index.round(2),
    "Crop_Yield_tha":       crop_yield.round(3),
    "Carbon_Sequestered_kgCO2e": carbon_sequestered.round(1),
})

out_path = Path("data")
out_path.mkdir(exist_ok=True)
df.to_csv(out_path / "soil_biochar_dataset.csv", index=False)

print(f"✅  Dataset saved → data/soil_biochar_dataset.csv  ({len(df)} rows)")
print(df.describe().round(2))
