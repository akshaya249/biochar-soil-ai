"""
PHASE 7 — Streamlit Web Dashboard
Soil Intelligence System: Biochar from Prosopis juliflora

Run with:  streamlit run 05_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Soil Intelligence System",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 AI-Based Soil Optimization System")
st.markdown(
    "**Biochar from *Prosopis juliflora* (Seemai Karuvelam) — Tamil Nadu**  \n"
    "Enter your soil parameters below to get smart recommendations."
)

st.divider()

# ── Load models ───────────────────────────────────────────────────────────────
MODEL_DIR = Path("models")

@st.cache_resource
def load_models():
    models = {}
    targets = [
        "Soil_Health_Index",
        "Crop_Yield_tha",
        "Carbon_Sequestered_kgCO2e",
    ]
    for t in targets:
        path = MODEL_DIR / f"{t}_best_model.pkl"
        if path.exists():
            models[t] = joblib.load(path)
    return models

models = load_models()

if not models:
    st.error("⚠️  No trained models found. Run `03_train_models.py` first.")
    st.stop()

# ── Sidebar — Input form ───────────────────────────────────────────────────────
st.sidebar.header("🧪 Soil Parameters")

pH          = st.sidebar.slider("Soil pH",              5.0, 8.5, 6.5, 0.1)
nitrogen    = st.sidebar.slider("Nitrogen (kg/ha)",     80,  400, 180, 5)
phosphorus  = st.sidebar.slider("Phosphorus (kg/ha)",   5,   80,  20,  1)
potassium   = st.sidebar.slider("Potassium (kg/ha)",    60,  320, 140, 5)
moisture    = st.sidebar.slider("Moisture (%)",         10,  65,  30,  1)
org_carbon  = st.sidebar.slider("Organic Carbon (%)",  0.2, 2.5, 0.6, 0.05)
ec          = st.sidebar.slider("EC (dS/m)",            0.1, 1.2, 0.5, 0.05)
biochar_pct = st.sidebar.select_slider(
    "Biochar Application (%)", options=[0, 2, 5, 10], value=5
)

predict_btn = st.sidebar.button("🚀 Analyse Soil", use_container_width=True)

# ── Crop suggestion logic ─────────────────────────────────────────────────────
def suggest_crop(shi: float, pH: float, moisture: float) -> str:
    if shi >= 75 and 6.0 <= pH <= 7.5:
        return "🌾 Rice / Wheat — Excellent soil conditions"
    elif shi >= 60 and moisture >= 30:
        return "🌽 Maize / Sorghum — Good moisture retention"
    elif shi >= 45:
        return "🫘 Groundnut / Pulses — Moderate conditions"
    else:
        return "🌿 Cover crop (Green manure) — Soil needs recovery"

# ── Biochar dosage recommendation ─────────────────────────────────────────────
def recommend_dosage(shi: float) -> str:
    if shi < 40:
        return "**10% biochar** — Severely degraded soil, maximum amendment needed"
    elif shi < 60:
        return "**5% biochar** — Moderate degradation, standard treatment"
    elif shi < 75:
        return "**2% biochar** — Mild improvement needed"
    else:
        return "**0–2% biochar** — Healthy soil, minimal amendment required"

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn or True:   # also run on first load with defaults
    features = [[pH, nitrogen, phosphorus, potassium, moisture, org_carbon, ec, biochar_pct]]
    feature_names = ["pH", "Nitrogen_kgha", "Phosphorus_kgha", "Potassium_kgha",
                     "Moisture_pct", "Organic_Carbon_pct", "EC_dSm", "Biochar_pct"]
    X_input = pd.DataFrame(features, columns=feature_names)

    shi    = models["Soil_Health_Index"].predict(X_input)[0]
    yield_ = models["Crop_Yield_tha"].predict(X_input)[0]
    carbon = models["Carbon_Sequestered_kgCO2e"].predict(X_input)[0]

    # ── Metrics row ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    shi_color = "🟢" if shi >= 70 else ("🟡" if shi >= 45 else "🔴")
    col1.metric(f"{shi_color} Soil Health Index", f"{shi:.1f} / 100")
    col2.metric("🌾 Est. Crop Yield", f"{yield_:.2f} t/ha")
    col3.metric("🌍 Carbon Sequestered", f"{carbon:.0f} kg CO₂e/ha")

    st.divider()

    # ── Recommendations ───────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("💡 Recommendations")
        st.markdown(f"**Biochar Dosage:** {recommend_dosage(shi)}")
        st.markdown(f"**Suggested Crop:** {suggest_crop(shi, pH, moisture)}")

        # Health gauge
        health_label = (
            "🟢 Healthy"   if shi >= 70 else
            "🟡 Moderate"  if shi >= 45 else
            "🔴 Degraded"
        )
        st.progress(int(shi), text=f"Soil Health: {health_label} ({shi:.1f}/100)")

    with col_b:
        st.subheader("🌍 Carbon Impact")
        co2_equiv = carbon
        cars_off   = co2_equiv / 4600   # avg car emits 4.6 t CO₂/year
        st.markdown(
            f"- **{co2_equiv:.0f} kg CO₂e** sequestered per hectare  \n"
            f"- Equivalent to taking **{cars_off:.2f} cars** off the road for 1 year  \n"
            f"- Biochar stability: ~100 years in soil"
        )

    st.divider()

    # ── Input summary table ───────────────────────────────────────────────────
    with st.expander("📋 View Input Parameters"):
        summary = {
            "Parameter": ["pH", "Nitrogen (kg/ha)", "Phosphorus (kg/ha)",
                          "Potassium (kg/ha)", "Moisture (%)",
                          "Organic Carbon (%)", "EC (dS/m)", "Biochar (%)"],
            "Value": [pH, nitrogen, phosphorus, potassium,
                      moisture, org_carbon, ec, biochar_pct]
        }
        st.table(pd.DataFrame(summary))

st.caption("Built for AI-Based Soil Optimization Research | Tamil Nadu Agricultural Context")
