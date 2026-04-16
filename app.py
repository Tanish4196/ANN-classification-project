from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model, Model

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# Constants for project paths
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "churn_ann_model.keras"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "feature_columns.json"

# ------------------ LOAD ------------------
@st.cache_resource
def load_all() -> Tuple[Model | None, Any | None, list[str] | None]:
    """Loads model, scaler, and features with error handling."""
    model: Model | None = None
    scaler: Any | None = None
    feature_cols: list[str] | None = None

    # Check if files exist before loading
    missing_files = []
    if not MODEL_PATH.exists(): missing_files.append("Model (churn_ann_model.keras)")
    if not SCALER_PATH.exists(): missing_files.append("Scaler (scaler.pkl)")
    if not FEATURES_PATH.exists(): missing_files.append("Features (feature_columns.json)")

    if missing_files:
        st.error(f"❌ Missing required artifact files in `artifacts/` directory:\n" + "\n".join(f"- {f}" for f in missing_files))
        st.info("💡 Please ensure all trained artifacts are placed in the `artifacts/` folder.")
        return None, None, None

    try:
        model = load_model(MODEL_PATH)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(FEATURES_PATH, "r") as f:
            feature_cols = json.load(f)
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        return None, None, None

    return model, scaler, feature_cols

# Initial load
model, scaler, feature_cols = load_all()

# ------------------ UI ------------------
st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a customer will **churn or stay** using ANN")

st.divider()

if not model or not scaler or not feature_cols:
    st.warning("⚠️ Application is currently unavailable due to missing or corrupted model artifacts.")
    st.stop()

# INPUTS
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", 300, 900, 600)
    age = st.slider("Age", 18, 100, 30)
    tenure = st.slider("Tenure (Years)", 0, 10, 3)
    balance = st.number_input("Balance", 0.0, 1000000.0, 50000.0)

with col2:
    num_products = st.slider("Number of Products", 1, 4, 2)
    has_cr_card = st.selectbox("Has Credit Card", [0, 1])
    is_active_member = st.selectbox("Is Active Member", [0, 1])
    estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

gender = st.selectbox("Gender", ["Male", "Female"])
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

# ------------------ PREPROCESS ------------------
def preprocess() -> npt.NDArray[Any]:
    """Prepares user input for model prediction."""
    input_dict = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "Gender": gender,
        "Geography": geography,
    }

    df = pd.DataFrame([input_dict])

    # Encoding
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    df = pd.get_dummies(df, columns=["Geography"])

    # Align columns with trained features
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_cols]

    # Scaling
    df_scaled = scaler.transform(df)
    return df_scaled

# ------------------ PREDICT ------------------
if st.button("🔮 Predict Churn"):
    with st.spinner("Analyzing customer data..."):
        data = preprocess()
        prediction = model.predict(data)[0][0]

    st.divider()
    st.subheader("📈 Prediction Result")

    churn_prob = float(prediction) * 100
    retain_prob = 100 - churn_prob

    c1, c2 = st.columns(2)
    c1.metric("Churn Probability", f"{churn_prob:.2f}%")
    c2.metric("Retention Probability", f"{retain_prob:.2f}%")

    if churn_prob > 50:
        st.error("⚠️ High Risk: Customer is likely to churn.")
        st.progress(churn_prob / 100)
    else:
        st.success("✅ Low Risk: Customer is likely to stay.")
        st.progress(retain_prob / 100)