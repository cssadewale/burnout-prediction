# 🧠 Employee Burnout Rate Predictor — NeuroWell Analytics

> **Predicting employee burnout before it becomes a crisis.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://burnout-predictor-adeagbo.streamlit.app)

---

## 🔗 Live App

👉 **[Launch the Burnout Rate Predictor](https://burnout-predictor-adeagbo.streamlit.app)**

---

## 👤 Author

**Adewale Samson Adeagbo**
*Data Scientist | Data Analyst | ML Engineer*

| Contact | Link |
|---------|------|
| 📧 Email | [buildingmyictcareer@gmail.com](mailto:buildingmyictcareer@gmail.com) |
| 🔗 LinkedIn | [linkedin.com/in/adewalesamsonadeagbo](https://linkedin.com/in/adewalesamsonadeagbo) |
| 🐙 GitHub | [github.com/adewalesamsonadeagbo](https://github.com/adewalesamsonadeagbo) |
| 📞 Phone | 08100866322 |

---

## 📌 Project Overview

**Client:** NeuroWell Analytics — a global leader in workplace well-being solutions
**Role:** Lead Data Scientist / ML Engineer
**Project Type:** Supervised Machine Learning — Regression

In today's fast-paced work environment, employee burnout silently erodes productivity,
morale, and organisational resilience — often before leadership has any visibility into
the problem.

This project builds and deploys a machine learning solution that **predicts employee
burnout rates** from historical HR data, enabling HR teams and senior leadership to
intervene proactively before burnout escalates into attrition or absenteeism.

**Target variable:** `burn_rate` — a continuous score from **0.0** (no burnout)
to **1.0** (complete burnout).

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Algorithm** | Gradient Boosting Regressor |
| **R² Score** | 0.855 |
| **RMSE** | 0.072 |
| **MAE** | 0.053 |
| **Training records** | 22,750 employees |
| **Validation strategy** | 80/20 split + 5-fold cross-validation |

---

## 🔑 Key Findings

| Finding | Detail |
|---------|--------|
| **Strongest predictor** | `mental_fatigue_score` (Pearson r = **+0.878**, feature importance = **85.9%**) |
| **Second strongest** | `resource_allocation` (r = **+0.810**, importance = **14.1%**) |
| **Most actionable lever** | WFH access — employees without WFH have **12.2pp higher** average burn rate |
| **Highest-risk subgroup** | Senior employees (Designation 4–5) without WFH access |
| **Company type** | Non-factor — only **0.18pp** difference between Service and Product |

---

## 🗂️ Project Workflow

| Step | Phase | Description |
|------|-------|-------------|
| 1 | Data Loading & Inspection | Loaded train (22,750 × 9) and test (12,250 × 8) datasets |
| 2 | Data Cleaning | Column standardisation, datetime conversion, median imputation, duplicate check |
| 3 | EDA | Univariate → Bivariate → Multivariate analysis with statistical insights |
| 4 | Feature Engineering | Derived `tenure_months`, IQR capping, One-Hot Encoding, MinMaxScaling |
| 5 | Model Development | Linear Regression, Random Forest, Gradient Boosting — tuned with GridSearchCV |
| 6 | Business Insights | 6 priority-labelled recommendations for NeuroWell Analytics clients |

---

## 📁 Repository Structure

```
burnout-predictor/
├── app.py                                          # Streamlit web application
├── requirements.txt                                # Python dependencies
├── best_burnout_prediction_model.joblib            # Trained Gradient Boosting model
└── Burn_Rate_Prediction_NeuroWell_Analytics.ipynb  # Full analysis notebook
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| ML Library | scikit-learn |
| Algorithm | Gradient Boosting Regressor |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| App Framework | Streamlit |
| Model Serialisation | joblib |
| Deployment | Streamlit Cloud |

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/adewalesamsonadeagbo/burnout-predictor.git
cd burnout-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

---

## 💼 Portfolio

This project is part of a growing portfolio of end-to-end ML projects deployed as
Streamlit applications. Each project follows a rigorous 6-step analytical framework:
Data Loading → Cleaning → EDA → Feature Engineering → Modelling → Business Insights.

**Other projects:**
- 🏦 Bank Churn Prediction — Gradient Boosting
- 🏥 Insurance Claim Prediction — Random Forest + SHAP
- 👔 Staff Promotion Prediction — Random Forest
- 💰 Income Level Prediction — Random Forest
- 📦 SwiftChain Delivery Delay Prediction — Multiclass Gradient Boosting

---

*Built with ❤️ by Adewale Samson Adeagbo · Lagos, Nigeria*
