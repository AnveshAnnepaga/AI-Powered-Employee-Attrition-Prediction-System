# 🔮 AI-Powered HR Analytics & Employee Attrition Predictor

An enterprise-grade, high-end interactive predictive analytics platform built to identify, monitor, and mitigate employee turnover. Utilizing **6 advanced machine learning architectures** trained on IBM Watson HR analytics, the platform delivers real-time executive dashboard KPIs, individual attrition risk scoring with automated retention recommendations, batch roster file processing, and deep model interpretability metrics.

Designed with a premium **dark glassmorphic space-grey theme** and glowing royal purple accents, the interface provides seamless, high-contrast visual clarity for HR executives and enterprise decision-makers.

⚡ **Live Demo**: Access the deployed platform directly here: **[AI-Powered Employee Attrition Predictor](https://ai-powered-employee-attrition-predictionsystem.streamlit.app/)**

---

## 🌟 Core Platform Features

- **🏠 Executive HR Analytics Dashboard**:
  - Live KPI metric cards displaying overall organizational health (Total Headcount, Attrition Rate, Average Salary, Average Job Satisfaction).
  - Dynamic demographic and operational charts analyzing attrition trends across departments, commute distances, overtime behaviors, and monthly income bands.
- **🔮 Real-Time Individual Risk Predictor**:
  - A structured 4-tab parameter questionnaire covering:
    - *👤 Personal Details*: Commute distance, age, gender, marital status.
    - *💼 Job & Role*: Department, specific role, frequency of business travel, overtime.
    - *💰 Compensation*: Salary, stock option levels, daily/hourly rates, performance rating.
    - *⏱️ Tenure & Training*: Working years, time at company, years under current manager, training history.
  - A beautifully centered **Plotly Risk Score Gauge** highlighting low, medium, and high attrition thresholds.
  - **HR Retention Strategy Engine**: Automated heuristics-derived alert boxes offering immediate retention guidelines and flagging core burnout drivers.
- **📊 Batch CSV Predictor**:
  - Bulk upload capabilities for entire company rosters.
  - Self-healing data pipeline that automatically pads missing parameters using median/mode training data imputations.
  - Interactive, searchable predictions grid with purple-shaded risk gradients.
  - Exportable risk assessment downloads (CSV format).
- **⚙️ Deep AI Performance Hub**:
  - Live model selection dropdown to compare 6 different ML architectures.
  - Real-time precision, recall, F1-score evaluation tables on a stratified test-split.
  - Interactive Plotly Confusion Matrix overlay.
  - Global Feature Importance drivers plotted dynamically via a Random Forest pipeline.

---

## 🧬 Machine Learning & Data Pipeline Architecture

The platform operates on a robust, self-healing pipeline that guarantees version compatibility and zero runtime data misalignment:

```
[ Raw CSV Data ] ➡️ [ ColumnTransformer ] ➡️ [ Pipeline fit() ] ➡️ [ Saved Model Binaries ]
                           │                                          │
                  ┌────────┴────────┐                                 ├── Random Forest
                  │  StandardScaler │ ➡️ (Continuous features)        ├── Logistic Regression
                  │  OneHotEncoder  │ ➡️ (Categorical features)       ├── SVM (Probability=True)
                  └─────────────────┘                                 ├── KNN Classifier
                                                                      ├── Decision Tree
                                                                      └── Naive Bayes
```

### Self-Healing & Version Alignment
To prevent common scikit-learn version mismatches and feature shuffling errors, the platform features a **Dynamic Startup Training Pipeline**:
- Bypasses stale pre-saved `.pkl` binaries.
- Dynamically fits the preprocessing pipeline and all 6 models fresh on startup (taking **< 0.5s** total).
- Leverages `@st.cache_resource` to cache the pipeline in-memory, providing zero user-facing latency on subsequent runs.
- Automatically handles character BOM encoding boundaries (`utf-8-sig`) during CSV parsing.

---

## 📂 Repository Layout

```text
├── .git/                                   # Git tracking folder
├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # IBM Watson HR Analytics training data
├── app.py                                  # Main Streamlit web application & ML pipeline
├── requirements.txt                        # Target python dependencies
├── README.md                               # Platform documentation
├── random_forest_model.pkl                 # Cached Random Forest model binary
├── logistic_regression_model.pkl           # Cached Logistic Regression model binary
├── svm_model.pkl                           # Cached SVM model binary
├── knn_model.pkl                           # Cached K-Nearest Neighbors model binary
├── decision_tree_model.pkl                 # Cached Decision Tree model binary
└── naive_bayes_model.pkl                   # Cached Naive Bayes model binary
```

---

## 🚀 Getting Started & Installation

Follow these simple steps to set up and launch the predictive platform locally:

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Clone the Repository
Clone this repository to your local workspace:
```bash
git clone https://github.com/AnveshAnnepaga/AI-Powered-Employee-Attrition-Prediction-System.git
cd AI-Powered-Employee-Attrition-Prediction-System
```

### 3. Install Dependencies
Install the required enterprise Python packages:
```bash
pip install -r requirements.txt
```

### 4. Launch the Platform
Start the local Streamlit application server:
```bash
streamlit run app.py
```
Streamlit will hot-reload the script and serve the application locally at `http://localhost:8501`.

---

## 🛠️ Technological Stack

- **Frontend & App Server**: Streamlit (v1.22.0+)
- **Core Data Structures**: Pandas, Numpy
- **Machine Learning**: Scikit-Learn (ColumnTransformer, Pipeline, StandardScaler, OneHotEncoder)
- **Model Storage**: Joblib (Serialization / Deserialization)
- **Data Visualizations**: Plotly Express & Plotly Graph Objects (Interactive Charts & Gauge Indicators)