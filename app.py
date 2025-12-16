# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Page config (cleaner UI)
# -----------------------------
st.set_page_config(
    page_title="Animal Adoption Prediction",
    page_icon="🐾",
    layout="centered"
)

# -----------------------------
# Load model + expected columns
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("gb_model.pkl")          # your trained model
    columns = joblib.load("gb_columns.pkl")      # list/Index of training dummy columns
    # Ensure columns is a plain list for safe reindex
    if not isinstance(columns, list):
        columns = list(columns)
    return model, columns

model, columns = load_artifacts()

# -----------------------------
# Helper: build model input row
# -----------------------------
def build_input_df(
    animal_type: str,
    age_months: float,
    sex: str,
    intake_condition: str,
    intake_type: str,
    season: str | None = None
) -> pd.DataFrame:
    data = {
        "Animal Type": animal_type,
        "Age_Months": float(age_months),
        "Sex": sex,
        "Intake Condition": intake_condition,
        "Intake Type": intake_type,
    }
    if season is not None:
        data["Season"] = season

    raw = pd.DataFrame([data])
    encoded = pd.get_dummies(raw, drop_first=True)
    encoded = encoded.reindex(columns=columns, fill_value=0)
    return encoded

def predict_probability(input_encoded: pd.DataFrame) -> float:
    # binary class probability for class=1
    proba = model.predict_proba(input_encoded)[0][1]
    return float(proba)

# -----------------------------
# UI
# -----------------------------
st.title("🐾 Animal Adoption Prediction Dashboard")
st.caption(
    "Enter intake-time information and get a predicted adoption probability. "
    "This is meant as a decision-support tool, not an automated gatekeeper."
)

with st.sidebar:
    st.header("Intake Inputs")

    animal_type = st.selectbox("Animal Type", ["DOG", "CAT"], index=0)

    # Age input: human-friendly (Years/Months), model uses months
    age_unit = st.radio("Age unit", ["Years", "Months"], horizontal=True, index=0)

    if age_unit == "Years":
        age_years = st.slider(
            "Age (years)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.5
        )
        age_months = age_years * 12.0
    else:
        age_months = st.slider(
            "Age (months)",
            min_value=0,
            max_value=240,
            value=24,
            step=1
        )

    sex = st.selectbox(
        "Sex / Sterilization Status",
        ["Male", "Female", "Neutered", "Spayed", "Unknown"],
        index=2
    )

    intake_condition = st.selectbox(
        "Intake Condition",
        ["NORMAL", "ILL MILD", "INJURED  MILD", "INJURED  MODERATE", "INJURED  SEVERE", "FERAL", "FRACTIOUS"],
        index=0
    )

    intake_type = st.selectbox(
        "Intake Type",
        ["STRAY", "OWNER SURRENDER", "PUBLIC ASSIST", "OTHER"],
        index=0
    )

    # Optional: season (if you keep it)
    season = None
    with st.expander("Advanced (optional)", expanded=False):
        use_season = st.checkbox("Include Season", value=False)
        if use_season:
            season = st.selectbox("Season of Intake", ["Spring", "Summer", "Fall", "Winter"], index=0)

    st.divider()
    predict_btn = st.button("Predict adoption probability", type="primary")


# -----------------------------
# Main output
# -----------------------------
if predict_btn:
    X_input = build_input_df(
        animal_type=animal_type,
        age_months=age_months,
        sex=sex,
        intake_condition=intake_condition,
        intake_type=intake_type,
        season=season
    )
    prob = predict_probability(X_input)

    # Display result nicely
    st.subheader("Predicted Adoption Probability")
    st.metric(label="Adoption probability", value=f"{prob*100:.2f}%")

    # Simple interpretation bucket
    if prob >= 0.70:
        band = "High"
        note = "High likelihood. Consider fast-track promotion / standard adoption flow."
    elif prob >= 0.40:
        band = "Medium"
        note = "Moderate likelihood. Consider normal workflow + targeted improvements if needed."
    else:
        band = "Low"
        note = "Lower likelihood. Consider additional support (medical/grooming/behavior assessment) and extra promotion."

    st.info(f"**Risk band:** {band}\n\n{note}")

    with st.expander("Show model input (debug)", expanded=False):
        st.write("This is the one-hot encoded row aligned to training columns.")
        st.dataframe(X_input)

else:
    st.subheader("How to use")
    st.write(
        "- Fill in the intake-time fields on the left.\n"
        "- Click **Predict adoption probability**.\n"
        "- Use the score as a **triage / ranking** signal (not an automatic decision)."
    )

st.divider()
st.caption(
    "Note: If you trained the model without certain features (e.g., Season), including them here may have no effect. "
    "Keep your report and dashboard features consistent."
)
