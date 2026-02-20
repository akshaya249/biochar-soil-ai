"""
AI-Based Soil Optimization System — Streamlit Dashboard
Auto-generates data and trains models if not found (works on Streamlit Cloud)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

st.set_page_config(page_title="Soil Intelligence System", page_icon="🌱", layout="wide")

FEATURES = ["pH", "Nitrogen_kgha", "Phosphorus_kgha", "Potassium_kgha",
            "Moisture_pct", "Organic_Carbon_pct", "EC_dSm", "Biochar_pct"]
TARGETS  = ["Soil_Health_Index", "Crop_Yield_tha", "Carbon_Sequestered_kgCO2e"]

@st.cache_data
def generate_data():
    np.random.seed(42)
    N = 500
    biochar_pct = np.random.choice([0, 2, 5, 10], size=N)
    improvement = biochar_pct / 10
    pH         = np.clip(np.random.normal(6.5, 0.6, N) + improvement * 0.4, 5.0, 8.5)
    nitrogen   = np.clip(np.random.normal(180, 40, N)  + improvement * 60,  80, 400)
    phosphorus = np.clip(np.random.normal(20,  8,  N)  + improvement * 15,  5,  80)
    potassium  = np.clip(np.random.normal(140, 30, N)  + improvement * 40,  60, 320)
    moisture   = np.clip(np.random.normal(30,  8,  N)  + improvement * 12,  10, 65)
    org_carbon = np.clip(np.random.normal(0.6, 0.2, N) + improvement * 0.5, 0.2, 2.5)
    ec         = np.clip(np.random.normal(0.5, 0.15,N) - improvement * 0.05,0.1, 1.2)
    pH_score   = 100 - np.abs(pH - 6.8) * 20
    soil_health_index = np.clip(
        0.20 * pH_score +
        0.20 * np.clip(nitrogen   / 400 * 100, 0, 100) +
        0.15 * np.clip(phosphorus / 80  * 100, 0, 100) +
        0.15 * np.clip(potassium  / 320 * 100, 0, 100) +
        0.15 * np.clip(moisture   / 65  * 100, 0, 100) +
        0.10 * np.clip(org_carbon / 2.5 * 100, 0, 100) +
        0.05 * (100 - np.clip(ec / 1.2 * 100, 0, 100))
        + np.random.normal(0, 3, N), 0, 100)
    crop_yield = np.clip(1.5 + soil_health_index * 0.04 + improvement * 0.8
                         + np.random.normal(0, 0.3, N), 0.5, 7.0)
    carbon_seq = np.clip(biochar_pct * 100 * 0.75 * 0.85 * (44/12)
                         + np.random.normal(0, 50, N), 0, None)
    return pd.DataFrame({
        "pH": pH.round(2), "Nitrogen_kgha": nitrogen.round(1),
        "Phosphorus_kgha": phosphorus.round(1), "Potassium_kgha": potassium.round(1),
        "Moisture_pct": moisture.round(1), "Organic_Carbon_pct": org_carbon.round(3),
        "EC_dSm": ec.round(3), "Biochar_pct": biochar_pct,
        "Soil_Health_Index": soil_health_index.round(2),
        "Crop_Yield_tha": crop_yield.round(3),
        "Carbon_Sequestered_kgCO2e": carbon_seq.round(1),
    })

@st.cache_resource
def train_models():
    df = generate_data()
    X  = df[FEATURES]
    models = {}
    for target in TARGETS:
        y    = df[target]
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("model", RandomForestRegressor(n_estimators=200, random_state=42))])
        pipe.fit(X, y)
        models[target] = pipe
    return models

with st.spinner("🔄 Loading AI models... (first load takes ~30 seconds)"):
    models = train_models()

st.title("🌱 AI-Based Soil Optimization System")
st.markdown(
    "**Biochar from *Prosopis juliflora* (Seemai Karuvelam) — Tamil Nadu**  \n"
    "Enter your soil parameters below to get smart recommendations."
)
st.divider()

st.sidebar.header("🧪 Soil Parameters")
pH          = st.sidebar.slider("Soil pH",             5.0, 8.5, 6.5, 0.1)
nitrogen    = st.sidebar.slider("Nitrogen (kg/ha)",    80,  400, 180, 5)
phosphorus  = st.sidebar.slider("Phosphorus (kg/ha)",  5,   80,  20,  1)
potassium   = st.sidebar.slider("Potassium (kg/ha)",   60,  320, 140, 5)
moisture    = st.sidebar.slider("Moisture (%)",        10,  65,  30,  1)
org_carbon  = st.sidebar.slider("Organic Carbon (%)", 0.2, 2.5, 0.6, 0.05)
ec          = st.sidebar.slider("EC (dS/m)",           0.1, 1.2, 0.5, 0.05)
biochar_pct = st.sidebar.select_slider("Biochar Application (%)", options=[0,2,5,10], value=5)

X_input = pd.DataFrame([[pH, nitrogen, phosphorus, potassium,
                          moisture, org_carbon, ec, biochar_pct]], columns=FEATURES)
shi    = models["Soil_Health_Index"].predict(X_input)[0]
yield_ = models["Crop_Yield_tha"].predict(X_input)[0]
carbon = models["Carbon_Sequestered_kgCO2e"].predict(X_input)[0]

shi_icon = "🟢" if shi >= 70 else ("🟡" if shi >= 45 else "🔴")
col1, col2, col3 = st.columns(3)
col1.metric(f"{shi_icon} Soil Health Index", f"{shi:.1f} / 100")
col2.metric("🌾 Est. Crop Yield",            f"{yield_:.2f} t/ha")
col3.metric("🌍 Carbon Sequestered",          f"{carbon:.0f} kg CO₂e/ha")

st.divider()

def suggest_crop(shi, pH, moisture):
    if shi >= 75 and 6.0 <= pH <= 7.5: return "🌾 Rice / Wheat — Excellent soil conditions"
    elif shi >= 60 and moisture >= 30:  return "🌽 Maize / Sorghum — Good moisture retention"
    elif shi >= 45:                     return "🫘 Groundnut / Pulses — Moderate conditions"
    else:                               return "🌿 Cover crop (Green manure) — Soil needs recovery"

def recommend_dosage(shi):
    if shi < 40:   return "**10% biochar** — Severely degraded soil"
    elif shi < 60: return "**5% biochar** — Moderate degradation"
    elif shi < 75: return "**2% biochar** — Mild improvement needed"
    else:          return "**0–2% biochar** — Healthy soil"

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("💡 Recommendations")
    st.markdown(f"**Biochar Dosage:** {recommend_dosage(shi)}")
    st.markdown(f"**Suggested Crop:** {suggest_crop(shi, pH, moisture)}")
    label = "🟢 Healthy" if shi >= 70 else ("🟡 Moderate" if shi >= 45 else "🔴 Degraded")
    st.progress(int(shi), text=f"Soil Health: {label} ({shi:.1f}/100)")

with col_b:
    st.subheader("🌍 Carbon Impact")
    st.markdown(
        f"- **{carbon:.0f} kg CO₂e** sequestered per hectare  \n"
        f"- Equivalent to taking **{carbon/4600:.2f} cars** off the road for 1 year  \n"
        f"- Biochar stability: ~100 years in soil"
    )

st.divider()
with st.expander("📋 View Input Parameters"):
    st.table(pd.DataFrame({
        "Parameter": ["pH","Nitrogen (kg/ha)","Phosphorus (kg/ha)","Potassium (kg/ha)",
                      "Moisture (%)","Organic Carbon (%)","EC (dS/m)","Biochar (%)"],
        "Value":     [pH, nitrogen, phosphorus, potassium, moisture, org_carbon, ec, biochar_pct]
    }))

st.caption("Built for AI-Based Soil Optimization Research | Tamil Nadu Agricultural Context")
