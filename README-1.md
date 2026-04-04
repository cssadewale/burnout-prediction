<div align="center">

# 🧠 Employee Burnout Rate Predictor

### Proactive Insights for Employee Well-being | NeuroWell Analytics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://adewale-burnout-prediction.streamlit.app)
&nbsp;
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
&nbsp;
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat&logo=scikit-learn&logoColor=white)
&nbsp;
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat&logo=streamlit&logoColor=white)
&nbsp;
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat)

<br/>

**[🚀 Launch Live App](https://adewale-burnout-prediction.streamlit.app)** &nbsp;|&nbsp; **[📓 View Notebook](./Burn_Rate_Prediction_NeuroWell_Analytics.ipynb)** &nbsp;|&nbsp; **[📊 View Visualisations](#-visualisations)**

<br/>

> *Can we accurately predict the rate at which an employee is burning out — and identify which factors drive it — so that organisations can act before it is too late?*

</div>

---

## 👤 About the Author

<table>
<tr><td><b>Name</b></td><td>Adewale Samson Adeagbo</td></tr>
<tr><td><b>Role</b></td><td>Data Scientist &nbsp;|&nbsp; Data Analyst &nbsp;|&nbsp; ML Engineer</td></tr>
<tr><td><b>Client</b></td><td>NeuroWell Analytics</td></tr>
<tr><td><b>Location</b></td><td>Lagos, Nigeria</td></tr>
<tr><td>📧 <b>Email</b></td><td><a href="mailto:buildingmyictcareer@gmail.com">buildingmyictcareer@gmail.com</a></td></tr>
<tr><td>🔗 <b>LinkedIn</b></td><td><a href="https://linkedin.com/in/adewalesamsonadeagbo">linkedin.com/in/adewalesamsonadeagbo</a></td></tr>
<tr><td>🐙 <b>GitHub</b></td><td><a href="https://github.com/adewalesamsonadeagbo">github.com/adewalesamsonadeagbo</a></td></tr>
<tr><td>📞 <b>Phone</b></td><td>08100866322 &nbsp;|&nbsp; 08094481488</td></tr>
</table>

---

## 📌 Project Overview

**NeuroWell Analytics** — a global leader in workplace well-being and productivity solutions — engaged me as **Lead Data Scientist / ML Engineer** to design and deploy a machine learning solution that predicts employee burnout rates from historical HR data.

The resulting model equips HR teams and senior leadership to **proactively intervene** before burnout escalates into attrition, absenteeism, or a full mental health crisis.

| | |
|---|---|
| **Project Type** | Supervised Machine Learning — Regression |
| **Target Variable** | `burn_rate` — continuous score from `0.0` (no burnout) to `1.0` (complete burnout) |
| **Training Dataset** | 22,750 employee records × 9 features |
| **Test Dataset** | 12,250 employee records × 8 features |
| **Best Model** | Gradient Boosting Regressor (tuned via GridSearchCV) |

---

## 📊 Model Performance

| Metric | Linear Regression | Random Forest | **Gradient Boosting (Selected)** |
|--------|:-----------------:|:-------------:|:--------------------------------:|
| **R² Score** | 0.8304 | 0.8347 | **0.855** |
| **RMSE** | 0.0781 | 0.0771 | **0.072** |
| **MAE** | 0.0569 | 0.0568 | **0.053** |

> The **Gradient Boosting Regressor** was selected as the best model across all three metrics. It explains **85.5% of burn rate variance** and predicts with a mean absolute error of just **0.053 points** on a 0–1 scale.

---

## 🔑 Key Findings

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **Mental fatigue is the primary burnout driver** | Pearson r = **+0.878** with `burn_rate`; accounts for **85.9%** of model feature importance |
| 2 | **Workload is the second largest driver** | `resource_allocation` r = **+0.810**; Q4 employees carry **2.7× more workload** than Q1 |
| 3 | **WFH access is the most actionable lever** | Employees without WFH have a **12.2 percentage point** higher average burn rate (0.518 vs 0.396) |
| 4 | **Seniority amplifies risk** | `designation` r = **+0.719**; Q4 employees average designation **3.25** vs **1.13** for Q1 |
| 5 | **Company type is a non-factor** | Only **0.18pp** difference between Service and Product companies |

---

## 📈 Visualisations

> Charts generated from the EDA and modelling pipeline. Click any image to enlarge.

### Target Variable Distribution
![Burn Rate Distribution](visualisations/burn_rate_distribution.png)

### Correlation Heatmap — Numerical Features
![Correlation Heatmap](visualisations/correlation_heatmap.png)

### Numerical Features vs Burn Rate
![Scatter Plots](visualisations/numerical_vs_burnrate.png)

### Categorical Features vs Burn Rate
![Box Plots](visualisations/categorical_vs_burnrate.png)

### Multivariate — Resource Allocation vs Mental Fatigue (coloured by Burn Rate)
![Multivariate Scatter](visualisations/resource_vs_fatigue_burnrate.png)

### Feature Importance — Gradient Boosting Regressor
![Feature Importance](visualisations/feature_importance.png)

### Predicted vs Actual Burn Rate
![Predicted vs Actual](visualisations/predicted_vs_actual.png)

---

## 🗂️ Project Workflow

```
Raw Data  (train: 22,750 × 9  |  test: 12,250 × 8)
    │
    ▼
STEP 1 ── Data Loading & Inspection
          Shape · dtypes · .info() · .describe() · unique values
    │
    ▼
STEP 2 ── Data Cleaning
          Column standardisation → datetime conversion →
          median imputation (3 cols) → duplicate check (0 found)
    │
    ▼
STEP 3 ── Exploratory Data Analysis
          Univariate (target → numerical → categorical)
          Bivariate  (Pearson correlation · scatter · box plots)
          Multivariate (pair plot · interaction scatter · heatmap)
    │
    ▼
STEP 4 ── Feature Engineering & Preprocessing
          tenure_months derivation → IQR outlier capping →
          One-Hot Encoding → MinMaxScaler → 80/20 train-val split
    │
    ▼
STEP 5 ── Model Development & Evaluation
          Linear Regression · Random Forest · Gradient Boosting
          → GridSearchCV tuning (108 combinations × 3-fold CV)
          → 5-fold cross-validation → Feature Importance analysis
    │
    ▼
STEP 6 ── Business Insights & Recommendations
          6 priority-labelled (HIGH / MEDIUM / LOW) actions
          for NeuroWell Analytics and its clients
    │
    ▼
Deployed Streamlit App  +  Serialised model (.joblib)
```

---

## 📁 Repository Structure

```
burnout-predictor/
│
├── app.py                                           # Streamlit web application
├── requirements.txt                                 # Python package dependencies
├── best_burnout_prediction_model.joblib             # Serialised trained model
├── Burn_Rate_Prediction_NeuroWell_Analytics.ipynb   # Full analysis notebook (202 cells)
├── README.md                                        # Project documentation
│
└── visualisations/
    ├── burn_rate_distribution.png                   # Target variable distribution
    ├── correlation_heatmap.png                      # Pearson correlation matrix
    ├── numerical_vs_burnrate.png                    # Scatter plots vs target
    ├── categorical_vs_burnrate.png                  # Box plots vs target
    ├── resource_vs_fatigue_burnrate.png             # Multivariate interaction
    ├── feature_importance.png                       # GB feature importances
    └── predicted_vs_actual.png                      # Model diagnostic plot
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10 |
| **Data Handling** | pandas, numpy |
| **Visualisation** | matplotlib, seaborn |
| **ML Framework** | scikit-learn |
| **Algorithm** | Gradient Boosting Regressor |
| **Hyperparameter Tuning** | GridSearchCV (108 combinations × 3-fold CV) |
| **App Framework** | Streamlit |
| **Model Serialisation** | joblib |
| **Deployment** | Streamlit Cloud |
| **Version Control** | Git / GitHub |

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

The app will open automatically at `http://localhost:8501`.

---

## 💼 Portfolio

This project is part of a portfolio of end-to-end ML solutions — each deployed as a live Streamlit application and built on a rigorous 6-step analytical framework: **Data Loading → Cleaning → EDA → Feature Engineering → Modelling → Business Insights.**

| # | Project | Task Type | Algorithm | Records |
|---|---------|-----------|-----------|---------|
| 1 | 🏥 Insurance Claim Prediction | Binary Classification | Random Forest + SHAP | 7,160 |
| 2 | 🏦 Bank Churn Prediction | Binary Classification | Gradient Boosting | 10,000 |
| 3 | 👔 Staff Promotion Prediction *(Yakub Trading Group)* | Binary Classification | Random Forest | 38,312 |
| 4 | 💰 Income Level Prediction | Binary Classification | Random Forest | 48,842 |
| 5 | **🧠 Burnout Rate Prediction *(NeuroWell Analytics)*** | **Regression** | **Gradient Boosting** | **22,750** |
| 6 | 📦 Delivery Delay Prediction *(SwiftChain)* | Multiclass Classification | Gradient Boosting | 15,549 |

---

<div align="center">

*Built with precision and purpose by **Adewale Samson Adeagbo** · Lagos, Nigeria*

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/adewalesamsonadeagbo)
&nbsp;&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github&logoColor=white)](https://github.com/adewalesamsonadeagbo)
&nbsp;&nbsp;
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=flat&logo=gmail&logoColor=white)](mailto:buildingmyictcareer@gmail.com)

</div>
