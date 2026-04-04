import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Burnout Rate Predictor | NeuroWell Analytics",
    page_icon="🧠",
    layout="centered"
)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("best_burnout_prediction_model.joblib")

model = load_model()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🧠 Employee Burnout Rate Predictor")
st.markdown(
    "**NeuroWell Analytics** · Predict employee burnout risk from HR data using Machine Learning"
)
st.markdown("---")

# ── Sidebar: About ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About This App")
    st.markdown(
        """
        This app predicts an employee's **burn rate** — a continuous score from
        **0.0** (no burnout) to **1.0** (complete burnout) — based on key HR metrics.

        **Model:** Gradient Boosting Regressor
        **R² Score:** 0.855
        **RMSE:** 0.072
        **MAE:** 0.053

        ---
        **Built by**

        Adewale Samson Adeagbo
        *Data Scientist | ML Engineer*

        📧 [buildingmyictcareer@gmail.com](mailto:buildingmyictcareer@gmail.com)
        🔗 [LinkedIn](https://linkedin.com/in/adewalesamsonadeagbo)
        🐙 [GitHub](https://github.com/adewalesamsonadeagbo)

        ---
        **Key Finding**

        Mental fatigue score drives **85.9%** of burnout predictions.
        Employees without WFH access have a **12.2pp** higher average burn rate.
        """
    )

# ── Input form ─────────────────────────────────────────────────────────────────
st.subheader("📋 Enter Employee Details")
st.caption("Adjust the inputs below and click Predict to get the burnout score.")

col1, col2 = st.columns(2)

with col1:
    designation = st.slider(
        "Designation Level",
        min_value=0.0, max_value=5.0, value=2.0, step=1.0,
        help="0 = Entry level, 5 = Most senior"
    )
    resource_allocation = st.slider(
        "Resource Allocation (Workload)",
        min_value=1.0, max_value=10.0, value=4.0, step=1.0,
        help="1 = Very light workload, 10 = Maximum workload"
    )
    mental_fatigue_score = st.slider(
        "Mental Fatigue Score",
        min_value=0.0, max_value=10.0, value=5.0, step=0.1,
        help="0 = No fatigue, 10 = Extreme fatigue (strongest predictor: r = 0.878)"
    )
    tenure_months = st.slider(
        "Tenure (months)",
        min_value=0.0, max_value=12.0, value=6.0, step=0.1,
        help="How long the employee has been with the organisation"
    )

with col2:
    gender = st.selectbox(
        "Gender",
        options=["Female", "Male"]
    )
    company_type = st.selectbox(
        "Company Type",
        options=["Product", "Service"]
    )
    wfh = st.selectbox(
        "WFH Setup Available",
        options=["No", "Yes"],
        help="Employees without WFH have 12.2pp higher average burn rate"
    )

# ── Prediction ─────────────────────────────────────────────────────────────────
st.markdown("---")

if st.button("🔮 Predict Burn Rate", use_container_width=True, type="primary"):

    # ── Encode categorical inputs ──────────────────────────────────────────────
    gender_male          = 1 if gender == "Male" else 0
    company_type_service = 1 if company_type == "Service" else 0
    wfh_yes              = 1 if wfh == "Yes" else 0

    # ── Build feature DataFrame (column order must match training exactly) ─────
    features = pd.DataFrame(
        [[
            designation,
            resource_allocation,
            mental_fatigue_score,
            tenure_months,
            gender_male,
            company_type_service,
            wfh_yes
        ]],
        columns=[
            'designation',
            'resource_allocation',
            'mental_fatigue_score',
            'tenure_months',
            'gender_Male',
            'company_type_Service',
            'wfh_setup_available_Yes'
        ]
    )

    # ── Apply MinMaxScaler using training-data bounds ──────────────────────────
    # Bounds derived from IQR-capped training data (fitted on X_train only)
    bounds = {
        'designation':             (0.0,  5.0),
        'resource_allocation':     (1.0, 10.0),
        'mental_fatigue_score':    (0.0, 10.0),
        'tenure_months':           (0.0, 12.0),
        'gender_Male':             (0.0,  1.0),
        'company_type_Service':    (0.0,  1.0),
        'wfh_setup_available_Yes': (0.0,  1.0),
    }

    features_scaled = features.copy()
    for col, (lo, hi) in bounds.items():
        features_scaled[col] = (features[col] - lo) / (hi - lo) if hi > lo else 0.0

    # ── Predict and clip to valid [0, 1] range ─────────────────────────────────
    raw_pred   = model.predict(features_scaled)[0]
    prediction = float(np.clip(raw_pred, 0.0, 1.0))

    # ── Assign risk tier ───────────────────────────────────────────────────────
    if prediction <= 0.33:
        tier, emoji = "Low Risk",      "🟢"
        colour      = "success"
        advice      = "Low burnout risk detected. Continue current workload and schedule a quarterly wellness check-in."
    elif prediction <= 0.55:
        tier, emoji = "Moderate Risk", "🟡"
        colour      = "warning"
        advice      = "Moderate burnout risk. Consider a workload review and monitor mental fatigue scores closely over the next 30 days."
    elif prediction <= 0.75:
        tier, emoji = "High Risk",     "🔴"
        colour      = "error"
        advice      = "High burnout risk. An immediate welfare check-in and workload reduction are strongly recommended."
    else:
        tier, emoji = "Critical Risk", "🚨"
        colour      = "error"
        advice      = "Critical burnout risk. Urgent HR intervention required. Consider immediate leave or role reassignment."

    # ── Display results ────────────────────────────────────────────────────────
    st.subheader("📊 Prediction Results")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(
            label="Predicted Burn Rate",
            value=f"{prediction:.3f}",
            delta=f"out of 1.0",
            delta_color="off"
        )
    with col_b:
        st.metric(
            label="Risk Tier",
            value=f"{emoji}  {tier}"
        )

    st.progress(prediction)

    if colour == "success":
        st.success(f"✅ **{tier}** — {advice}")
    elif colour == "warning":
        st.warning(f"⚠️ **{tier}** — {advice}")
    else:
        st.error(f"🔴 **{tier}** — {advice}")

    # ── Feature contribution context ───────────────────────────────────────────
    with st.expander("📌 What's driving this prediction?"):
        st.markdown(
            f"""
            | Factor | Your Input | Impact |
            |--------|-----------|--------|
            | Mental Fatigue Score | **{mental_fatigue_score:.1f} / 10** | 🔴 Highest (85.9% of model weight) |
            | Resource Allocation | **{resource_allocation:.0f} / 10** | 🟠 High (14.1% of model weight) |
            | WFH Available | **{wfh}** | {'🟢 Reduces burnout risk' if wfh == 'Yes' else '🔴 No WFH → higher risk (+12.2pp avg)'} |
            | Designation Level | **{designation:.0f} / 5** | {'🟡 Senior roles carry compounding risk' if designation >= 4 else '🟢 Lower seniority'} |
            | Gender | **{gender}** | {'🟡 Slightly higher avg burn rate' if gender == 'Male' else '🟢 Slightly lower avg burn rate'} |
            | Company Type | **{company_type}** | 🟢 Minimal effect (0.18pp difference) |
            | Tenure | **{tenure_months:.1f} months** | ⚪ Low importance in current model |
            """
        )
        st.caption(
            "Impact weights are based on Gradient Boosting feature importances "
            "and EDA findings from the NeuroWell Analytics project."
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Model: Gradient Boosting Regressor · R² = 0.855 · RMSE = 0.072 · MAE = 0.053 · "
    "Training data: 22,750 employee records | "
    "Built by **Adewale Samson Adeagbo** — Data Scientist / ML Engineer · "
    "[LinkedIn](https://linkedin.com/in/adewalesamsonadeagbo) · "
    "[GitHub](https://github.com/adewalesamsonadeagbo) · "
    "[buildingmyictcareer@gmail.com](mailto:buildingmyictcareer@gmail.com)"
)
