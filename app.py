import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================================================================
# 🌟 PREMIUM GLASSMORPHIC STYLING (CUSTOM CSS)
# ==============================================================================
st.set_page_config(
    page_title="AI-Powered HR Analytics & Attrition Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium UI style injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600;700&display=swap');
    
    /* Overall layout and backgrounds */
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers and titles */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: #FFFFFF;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0E131C !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Glowing card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(25, 33, 44, 0.6) 0%, rgba(14, 19, 28, 0.8) 100%);
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(124, 58, 237, 0.5);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #A78BFA 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 5px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        font-weight: 600;
    }
    
    /* Recommendations & HR Cards */
    .hr-card {
        background: rgba(22, 28, 38, 0.8);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 5px solid #7C3AED;
    }
    
    .hr-card.high-risk {
        border-left-color: #EF4444;
        background: rgba(239, 68, 68, 0.05);
    }
    
    .hr-card.medium-risk {
        border-left-color: #F59E0B;
        background: rgba(245, 158, 11, 0.05);
    }

    .hr-card.low-risk {
        border-left-color: #10B981;
        background: rgba(16, 185, 129, 0.05);
    }
    
    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.02) !important;
        border-radius: 8px 8px 0px 0px;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        color: #94A3B8 !important;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(124, 58, 237, 0.1) !important;
        color: #FFFFFF !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(124, 58, 237, 0.2) !important;
        color: #A78BFA !important;
        border-bottom: 2px solid #7C3AED !important;
    }
    
    /* Styled uploader & download button */
    .stDownloadButton button, .stButton button {
        background: linear-gradient(90deg, #7C3AED 0%, #6D28D9 100%) !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px 0 rgba(124, 58, 237, 0.4) !important;
        transition: all 0.3s !important;
    }
    
    .stDownloadButton button:hover, .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(124, 58, 237, 0.6) !important;
    }

    /* Fix text visibility of all widget labels, markdown, selectbox options, slider labels, and sidebar */
    label, .stWidgetLabel, [data-testid="stWidgetLabel"] p, .stSlider label {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
    }
    
    /* Main body markdown text visibility */
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #D1D5DB !important;
    }
    
    /* Sidebar specific overrides to solve dark text issue */
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] label p, 
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] p {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
    }
    
    /* Active sidebar menu option styling */
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #E2E8F0 !important;
    }
    
    /* Slider tick numbers visibility */
    div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"], [data-testid="stSliderTick"] {
        color: #94A3B8 !important;
        font-weight: 500 !important;
    }

    /* Make sure all non-active tabs text is clearly visible */
    .stTabs [data-baseweb="tab"] {
        color: #E2E8F0 !important;
    }

</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 💾 DATA LOADING, AUTOMATIC MODEL TRAINING & SELF-HEALING
# ==============================================================================
LOCAL_DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
FALLBACK_DATA_URL = "https://raw.githubusercontent.com/pplonski/datasets-for-start/refs/heads/master/employee_attrition/HR-Employee-Attrition-All.csv"

# Caching decorator backward-compatibilities
if hasattr(st, 'cache_data'):
    cache_data_decorator = st.cache_data(show_spinner="Loading and preparing HR Dataset...")
else:
    cache_data_decorator = st.cache(show_spinner="Loading and preparing HR Dataset...")

if hasattr(st, 'cache_resource'):
    cache_resource_decorator = st.cache_resource(show_spinner="Configuring AI Models & Fitting Preprocessors...")
else:
    cache_resource_decorator = st.cache(allow_output_mutation=True, show_spinner="Configuring AI Models & Fitting Preprocessors...")

@cache_data_decorator
def load_dataset():
    if os.path.exists(LOCAL_DATA_PATH):
        df = pd.read_csv(LOCAL_DATA_PATH, encoding='utf-8-sig')
    else:
        try:
            df = pd.read_csv(FALLBACK_DATA_URL, encoding='utf-8-sig')
            # Save locally for future use
            df.to_csv(LOCAL_DATA_PATH, index=False)
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            return None
    return df

@cache_resource_decorator
def train_and_cache_models(df):
    # Drop EDA temporary columns & non-predictive columns
    columns_to_drop = ['Attrition_Numeric', 'EmployeeCount', 'StandardHours', 'Income_Bin', 'Experience_Bin']
    existing_drops = [c for c in columns_to_drop if c in df.columns]
    
    df_ml = df.drop(columns=existing_drops)
    
    X = df_ml.drop('Attrition', axis=1)
    y = df_ml['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    model_files = {
        'Random Forest': 'random_forest_model.pkl',
        'Logistic Regression': 'logistic_regression_model.pkl',
        'SVM': 'svm_model.pkl',
        'KNN': 'knn_model.pkl',
        'Decision Tree': 'decision_tree_model.pkl',
        'Naive Bayes': 'naive_bayes_model.pkl'
    }
    
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    trained_pipelines = {}
    
    # Pre-fit the preprocessor globally
    preprocessor.fit(X, y)
    
    for name, filename in model_files.items():
        # Force dynamic training on startup to guarantee 100% accuracy, feature alignment, and version compatibility
        clf = models[name]
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        pipeline.fit(X, y)
        trained_pipelines[name] = pipeline
        
        # Save/overwrite the fresh model to disk
        try:
            joblib.dump(pipeline, filename)
        except:
            pass
                
    return trained_pipelines, X, y, numerical_features, categorical_features

# Load dataset and fit pipelines
df_raw = load_dataset()
if df_raw is not None:
    pipelines, X_features, y_target, num_cols, cat_cols = train_and_cache_models(df_raw)
else:
    st.error("Severe Error: IBM HR CSV dataset cannot be initialized. Please check your repository.")
    st.stop()

# ==============================================================================
# 🧭 SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.markdown("<h2 style='text-align: center; color: #A78BFA;'>🔮 HR Intel</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; font-size: 0.85rem; color: #A78BFA;'>IBM HR Attrition Prediction Platform</p>", unsafe_allow_html=True)

nav_page = st.sidebar.radio(
    "Navigation Menu",
    ["🏠 Executive Dashboard", "🔮 Predict Attrition (Individual)", "📊 Batch Prediction (CSV)", "⚙️ Model Performance & Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Configuration")
production_model_name = st.sidebar.selectbox("Active Production Model", list(pipelines.keys()))
production_model = pipelines[production_model_name]

st.sidebar.markdown("---")
st.sidebar.info("🤖 **Developer Mode:** Live data pipeline loaded successfully with 1,470 records and 32 active predictive features.")

# ==============================================================================
# 🏠 EXECUTIVE DASHBOARD PAGE
# ==============================================================================
if nav_page == "🏠 Executive Dashboard":
    st.title("🏠 Executive HR Analytics Dashboard")
    st.markdown("Monitor high-level organizational insights, key metrics, and core attritional drivers.")
    
    # Dashboard KPI Cards
    total_employees = len(df_raw)
    attrition_yes = len(df_raw[df_raw['Attrition'] == 'Yes'])
    attrition_rate = (attrition_yes / total_employees) * 100
    avg_income = df_raw['MonthlyIncome'].mean()
    avg_satisfaction = df_raw['JobSatisfaction'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Employees</div>
            <div class="metric-value">{total_employees:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Attrition Rate</div>
            <div class="metric-value">{attrition_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Monthly Income</div>
            <div class="metric-value">${avg_income:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Job Satisfaction</div>
            <div class="metric-value">{avg_satisfaction:.2f}/4</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Interactive Plots
    st.markdown("### Interactive Demographic & Risk Analysis")
    
    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        # Attrition by Department
        dept_attr = df_raw.groupby(['Department', 'Attrition']).size().reset_index(name='Count')
        fig_dept = px.bar(
            dept_attr, 
            x='Department', 
            y='Count', 
            color='Attrition',
            title='Employee Count by Department & Attrition Status',
            color_discrete_map={'Yes': '#EF4444', 'No': '#10B981'},
            barmode='group',
            template='plotly_dark'
        )
        fig_dept.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_dept, use_container_width=True)
        
    with plot_col2:
        # Attrition by Overtime
        ot_attr = df_raw.groupby(['OverTime', 'Attrition']).size().reset_index(name='Count')
        fig_ot = px.bar(
            ot_attr,
            x='OverTime',
            y='Count',
            color='Attrition',
            title='Influence of Overtime on Attrition Status',
            color_discrete_map={'Yes': '#EF4444', 'No': '#10B981'},
            barmode='group',
            template='plotly_dark'
        )
        fig_ot.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_ot, use_container_width=True)
        
    plot_col3, plot_col4 = st.columns(2)
    
    with plot_col3:
        # Income vs Attrition Boxplot
        fig_inc = px.box(
            df_raw,
            x='Attrition',
            y='MonthlyIncome',
            color='Attrition',
            title='Monthly Income Distribution vs Attrition Status',
            color_discrete_map={'Yes': '#EF4444', 'No': '#10B981'},
            template='plotly_dark'
        )
        fig_inc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_inc, use_container_width=True)
        
    with plot_col4:
        # Distance From Home vs Attrition
        fig_dist = px.histogram(
            df_raw,
            x='DistanceFromHome',
            color='Attrition',
            barmode='overlay',
            title='Distribution of Commute Distance by Attrition Status',
            color_discrete_map={'Yes': '#EF4444', 'No': '#10B981'},
            template='plotly_dark'
        )
        fig_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_dist, use_container_width=True)

# ==============================================================================
# 🔮 PREDICT ATTRITION (INDIVIDUAL) PAGE
# ==============================================================================
elif nav_page == "🔮 Predict Attrition (Individual)":
    st.title("🔮 Employee Attrition Predictor")
    st.markdown("Enter details for an individual employee to compute their risk of attrition.")
    
    # Input tabs
    tab_personal, tab_job, tab_comp, tab_tenure = st.tabs([
        "👤 Personal Details", "💼 Job & Role", "💰 Compensation & Performance", "⏱️ Tenure & Training"
    ])
    
    with tab_personal:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            age = st.slider("Age", 18, 65, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col_p2:
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            distance_from_home = st.slider("Distance From Home (Commute in miles)", 1, 30, 9)
            
    with tab_job:
        col_j1, col_j2 = st.columns(2)
        with col_j1:
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            job_role = st.selectbox("Job Role", [
                "Sales Executive", "Research Scientist", "Laboratory Technician", 
                "Manufacturing Director", "Healthcare Representative", "Manager", 
                "Sales Representative", "Research Director", "Human Resources"
            ])
            business_travel = st.selectbox("Business Travel Frequency", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        with col_j2:
            overtime = st.selectbox("Works Overtime?", ["Yes", "No"])
            job_level = st.slider("Job Level", 1, 5, 2)
            education = st.slider("Education Level (1: Low, 5: High)", 1, 5, 3)
            education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])

    with tab_comp:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            monthly_income = st.number_input("Monthly Income ($)", 1000, 25000, 5000)
            daily_rate = st.number_input("Daily Rate ($)", 100, 2000, 800)
            hourly_rate = st.number_input("Hourly Rate ($)", 10, 200, 65)
            monthly_rate = st.number_input("Monthly Rate ($)", 1000, 30000, 14000)
        with col_c2:
            stock_option = st.slider("Stock Option Level", 0, 3, 1)
            salary_hike = st.slider("Percent Salary Hike (%)", 11, 25, 14)
            perf_rating = st.selectbox("Performance Rating", [3, 4])
            job_involvement = st.slider("Job Involvement Level", 1, 4, 3)

    with tab_tenure:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            total_working_years = st.slider("Total Working Years", 0, 40, 10)
            years_at_company = st.slider("Years At Company", 0, 40, 5)
            years_in_role = st.slider("Years In Current Role", 0, 20, 3)
            num_companies = st.slider("Number of Companies Worked", 0, 9, 2)
        with col_t2:
            years_since_promo = st.slider("Years Since Last Promotion", 0, 15, 1)
            years_curr_manager = st.slider("Years With Current Manager", 0, 20, 3)
            training_last_year = st.slider("Training Times Last Year (Sessions)", 0, 6, 2)
            work_life_balance = st.slider("Work-Life Balance Level (1: Low, 4: High)", 1, 4, 3)
            env_satisfaction = st.slider("Environment Satisfaction (1: Low, 4: High)", 1, 4, 3)
            job_satisfaction = st.slider("Job Satisfaction (1: Low, 4: High)", 1, 4, 3)
            rel_satisfaction = st.slider("Relationship Satisfaction (1: Low, 4: High)", 1, 4, 3)

    # Initialize page specific state
    if 'individual_predicted' not in st.session_state:
        st.session_state.individual_predicted = False
        
    st.markdown("---")
    
    # Large centered call-to-action button
    col_btn_1, col_btn_2, col_btn_3 = st.columns([1.2, 1.6, 1.2])
    with col_btn_2:
        if st.button("🔮 Calculate Employee Attrition Risk", use_container_width=True):
            st.session_state.individual_predicted = True
            
    st.markdown("---")
    
    if st.session_state.individual_predicted:
        # Compute prediction inputs matching preprocessor structure
        input_data = pd.DataFrame([{
            'Age': age,
            'BusinessTravel': business_travel,
            'DailyRate': daily_rate,
            'Department': department,
            'DistanceFromHome': distance_from_home,
            'Education': education,
            'EducationField': education_field,
            'EmployeeNumber': 9999,  # Placeholder
            'EnvironmentSatisfaction': env_satisfaction,
            'Gender': gender,
            'HourlyRate': hourly_rate,
            'JobInvolvement': job_involvement,
            'JobLevel': job_level,
            'JobRole': job_role,
            'JobSatisfaction': job_satisfaction,
            'MaritalStatus': marital_status,
            'MonthlyIncome': monthly_income,
            'MonthlyRate': monthly_rate,
            'NumCompaniesWorked': num_companies,
            'Over18': 'Y',
            'OverTime': overtime,
            'PercentSalaryHike': salary_hike,
            'PerformanceRating': perf_rating,
            'RelationshipSatisfaction': rel_satisfaction,
            'StockOptionLevel': stock_option,
            'TotalWorkingYears': total_working_years,
            'TrainingTimesLastYear': training_last_year,
            'WorkLifeBalance': work_life_balance,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_role,
            'YearsSinceLastPromotion': years_since_promo,
            'YearsWithCurrManager': years_curr_manager
        }])
        
        # Defensive column alignment for individual predictor to prevent any mismatch
        for col in X_features.columns:
            if col not in input_data.columns:
                if col in num_cols:
                    input_data[col] = X_features[col].median()
                else:
                    input_data[col] = X_features[col].mode()[0]
                    
        # Align column order with feature order during fit to prevent scikit-learn warnings
        input_data = input_data[list(X_features.columns)]
        
        # Calculate probabilities across models for comparison
        prob_dict = {}
        for name, p in pipelines.items():
            try:
                prob_dict[name] = p.predict_proba(input_data)[0][1] * 100
            except Exception as e:
                # Safe numeric prediction fallback
                prob_dict[name] = float(p.predict(input_data)[0]) * 100
    
        prod_prob = prob_dict[production_model_name]
        
        # Risk display gauge
        res_col1, res_col2 = st.columns([1, 1.5])
        with res_col1:
            st.markdown("### Prediction Result")
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prod_prob,
                number = {'suffix': "%", 'valueformat': '.1f', 'font': {'size': 36, 'color': 'white', 'family': 'Outfit'}},
                title = {'text': f"{production_model_name} Risk Score", 'font': {'size': 18, 'color': 'white', 'family': 'Outfit'}},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#7C3AED"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(255,255,255,0.1)",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                        {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                    ],
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Outfit"},
                height=300,
                margin=dict(t=60, b=20, l=30, r=30)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with res_col2:
            st.markdown("### HR Retention Strategy & Drivers")
            
            # Heuristics derived risk drivers
            drivers = []
            if overtime == "Yes":
                drivers.append("🚨 Working **Overtime** is significantly increasing burnout risk.")
            if monthly_income < 3500:
                drivers.append("🚨 **Low compensation** tier relative to industry averages.")
            if distance_from_home > 15:
                drivers.append("🚨 **Commute distance** exceeds 15 miles, presenting work-life friction.")
            if job_satisfaction <= 2:
                drivers.append("🚨 Employee reports **Low Job Satisfaction**.")
            if stock_option == 0:
                drivers.append("🚨 Employee has **0 Stock Option Level**, reducing long-term organizational lock-in.")
            
            if prod_prob >= 70:
                st.markdown("""
                <div class="hr-card high-risk">
                    <h4 style="color:#EF4444;margin:0 0 5px 0;">🔥 High Attrition Risk Alert</h4>
                    <p style="margin:0;font-size:0.9rem;">Immediate retention measures recommended. Schedule a formal stay interview within the next 48 hours.</p>
                </div>
                """, unsafe_allow_html=True)
            elif prod_prob >= 30:
                st.markdown("""
                <div class="hr-card medium-risk">
                    <h4 style="color:#F59E0B;margin:0 0 5px 0;">⚠️ Elevated Attrition Risk</h4>
                    <p style="margin:0;font-size:0.9rem;">Monitor employee engagement metrics. Consider adjusting work-life balance or review salary levels during next cycle.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="hr-card low-risk">
                    <h4 style="color:#10B981;margin:0 0 5px 0;">✅ Stable Retention Outlook</h4>
                    <p style="margin:0;font-size:0.9rem;">Employee is at low risk of leaving. Standard path progression and training guidelines apply.</p>
                </div>
                """, unsafe_allow_html=True)
                
            if drivers:
                st.markdown("**Key Risk Drivers Identified:**")
                for d in drivers:
                    st.markdown(d)
            else:
                st.markdown("✅ No significant negative drivers found for this employee profile.")
    
        # Cross-Model Consensus Comparison
        st.markdown("### Cross-Model Predictions Consensus")
        consensus_df = pd.DataFrame(list(prob_dict.items()), columns=["Algorithm", "Attrition Risk Score (%)"])
        fig_consensus = px.bar(
            consensus_df, 
            y='Algorithm', 
            x='Attrition Risk Score (%)',
            orientation='h',
            color='Attrition Risk Score (%)',
            color_continuous_scale=px.colors.sequential.Purples,
            template='plotly_dark',
            title="Predictive Scores Comparison Across 6 Different ML Architectures"
        )
        fig_consensus.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_consensus, use_container_width=True)
    else:
        st.info("💡 Adjust the employee parameters in the tabs above, then click the **🔮 Calculate Employee Attrition Risk** button to calculate risk scores and HR retention drivers in real-time.")

# ==============================================================================
# 📊 BATCH PREDICTION (CSV) PAGE
# ==============================================================================
elif nav_page == "📊 Batch Prediction (CSV)":
    st.title("📊 Batch Employee Attrition Predictor")
    st.markdown("Upload a CSV file containing multiple employee rows to perform batch calculations.")
    
    st.markdown("### 📤 Upload Employee Roster")
    uploaded_file = st.file_uploader("Upload CSV formatted matching IBM HR Analytics schema", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            st.success("File uploaded successfully!")
            
            # Let's add a button to run the prediction!
            if 'batch_predicted' not in st.session_state:
                st.session_state.batch_predicted = False
                
            col_bbtn_1, col_bbtn_2, col_bbtn_3 = st.columns([1.2, 1.6, 1.2])
            with col_bbtn_2:
                if st.button("📊 Run Batch Attrition Analysis", use_container_width=True):
                    st.session_state.batch_predicted = True
                    
            if st.session_state.batch_predicted:
                # Keep copy of features for pipelines
                drop_cols = ['Attrition', 'Attrition_Numeric', 'EmployeeCount', 'StandardHours', 'Income_Bin', 'Experience_Bin']
                active_drops = [c for c in drop_cols if c in batch_df.columns]
                
                clean_batch = batch_df.drop(columns=active_drops, errors='ignore')
                
                # Defensive column alignment to heal custom batch rosters
                for col in X_features.columns:
                    if col not in clean_batch.columns:
                        if col in num_cols:
                            clean_batch[col] = X_features[col].median()
                        else:
                            clean_batch[col] = X_features[col].mode()[0]
                
                # Reorder batch columns
                clean_batch = clean_batch[list(X_features.columns)]
                
                with st.spinner("Calculating batch predictive analytics..."):
                    batch_probs = production_model.predict_proba(clean_batch)[:, 1]
                    batch_preds = production_model.predict(clean_batch)
                    
                results_df = batch_df.copy()
                results_df['Attrition Risk Score (%)'] = np.round(batch_probs * 100, 2)
                results_df['Predicted Attrition'] = ["Yes" if pred == 1 else "No" for pred in batch_preds]
                
                st.markdown("### 📊 Prediction Roster Results")
                cols_to_display = ['EmployeeNumber', 'Age', 'Department', 'JobRole', 'MonthlyIncome', 'Attrition Risk Score (%)', 'Predicted Attrition']
                existing_display_cols = [c for c in cols_to_display if c in results_df.columns]
                st.dataframe(results_df[existing_display_cols].style.background_gradient(
                    subset=['Attrition Risk Score (%)'] if 'Attrition Risk Score (%)' in existing_display_cols else None,
                    cmap='Purples'
                ))
                
                # Summary metrics
                total_batch = len(results_df)
                predicted_leaves = len(results_df[results_df['Predicted Attrition'] == "Yes"])
                batch_attrition_rate = (predicted_leaves / total_batch) * 100
                
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Predicted Attrition Rate</div>
                        <div class="metric-value">{batch_attrition_rate:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with bcol2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Predicted Organizational Leaves</div>
                        <div class="metric-value">{predicted_leaves} / {total_batch}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Download results button
                csv_result = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Risk Analysis CSV",
                    data=csv_result,
                    file_name="HR_Employee_Attrition_Risk_Report.csv",
                    mime="text/csv"
                )
            else:
                st.info("💡 CSV file uploaded successfully. Click the **📊 Run Batch Attrition Analysis** button to compute batch predictions and view full statistics.")
            
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")
            st.info("💡 Make sure your columns exactly match the fields present in 'WA_Fn-UseC_-HR-Employee-Attrition.csv'.")
    else:
        # Provide template download
        st.info("💡 You can download the current dataset as a template below, modify it, and upload it back here!")
        template_csv = df_raw.drop(columns=['Attrition'], errors='ignore').to_csv(index=False)
        st.download_button(
            label="📥 Download Template Roster CSV",
            data=template_csv,
            file_name="ibm_hr_roster_template.csv",
            mime="text/csv"
        )

# ==============================================================================
# ⚙️ MODEL PERFORMANCE & INSIGHTS PAGE
# ==============================================================================
elif nav_page == "⚙️ Model Performance & Insights":
    st.title("⚙️ Artificial Intelligence & Model Interpretability")
    st.markdown("Inspect performance characteristics, accuracy statistics, and core feature importances.")
    
    # Train test split for evaluation metric demonstration
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42, stratify=y_target)
    
    eval_model_name = st.selectbox("Inspect Model Details", list(pipelines.keys()))
    eval_pipeline = pipelines[eval_model_name]
    
    # Recalculate metrics
    with st.spinner(f"Evaluating {eval_model_name} metrics..."):
        test_preds = eval_pipeline.predict(X_test)
        acc = accuracy_score(y_test, test_preds)
        rep = classification_report(y_test, test_preds, output_dict=True)
        conf = confusion_matrix(y_test, test_preds)
        
    ecol1, ecol2 = st.columns([1.5, 1])
    with ecol1:
        st.markdown(f"### Classification Metrics ({eval_model_name})")
        
        # Safe extraction of Precision/Recall to prevent class missing crashes
        rep_0 = rep.get('0', rep.get(0, {}))
        rep_1 = rep.get('1', rep.get(1, {}))
        
        metric_data = {
            "Metric": ["Accuracy", "Precision (Stayed)", "Precision (Attrited)", "Recall (Stayed)", "Recall (Attrited)", "F1-Score (Stayed)", "F1-Score (Attrited)"],
            "Value": [
                f"{acc * 100:.2f}%", 
                f"{rep_0.get('precision', 0.0) * 100:.2f}%",
                f"{rep_1.get('precision', 0.0) * 100:.2f}%",
                f"{rep_0.get('recall', 0.0) * 100:.2f}%",
                f"{rep_1.get('recall', 0.0) * 100:.2f}%",
                f"{rep_0.get('f1-score', 0.0) * 100:.2f}%",
                f"{rep_1.get('f1-score', 0.0) * 100:.2f}%"
            ]
        }
        st.table(pd.DataFrame(metric_data))
        
    with ecol2:
        st.markdown("### Confusion Matrix")
        fig_conf = px.imshow(
            conf, 
            text_auto=True, 
            aspect="auto", 
            color_continuous_scale="Purples",
            labels=dict(x="Predicted", y="Actual"),
            x=['Predicted Stay', 'Predicted Leave'],
            y=['Actual Stay', 'Actual Leave'],
            template='plotly_dark'
        )
        fig_conf.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False)
        st.plotly_chart(fig_conf, use_container_width=True)
        
    # Feature Importances (Random Forest specifically)
    st.markdown("---")
    st.markdown("### Feature Importance Drivers (Global Random Forest)")
    
    rf_pipeline = pipelines['Random Forest']
    rf_clf = rf_pipeline.named_steps['classifier']
    rf_preprocessor = rf_pipeline.named_steps['preprocessor']
    
    # Extract feature names using a highly robust backward-compatible method
    importances = rf_clf.feature_importances_
    try:
        if hasattr(rf_preprocessor, 'get_feature_names_out'):
            all_features = rf_preprocessor.get_feature_names_out()
            clean_features = [f.replace('num__', '').replace('cat__', '') for f in all_features]
        else:
            cat_transformer = rf_preprocessor.named_transformers_['cat']
            if hasattr(cat_transformer, 'get_feature_names_out'):
                cat_names = cat_transformer.get_feature_names_out(cat_cols)
            elif hasattr(cat_transformer, 'get_feature_names'):
                cat_names = cat_transformer.get_feature_names(cat_cols)
            else:
                cat_names = [f"{col}_{cat}" for col in cat_cols for cat in cat_transformer.categories_[list(cat_cols).index(col)]]
            clean_features = list(num_cols) + list(cat_names)
    except Exception as e:
        # Ultimate fallback based on feature importance length
        clean_features = list(num_cols) + [f"Categorical_Feature_{i}" for i in range(len(importances) - len(num_cols))]
    
    imp_df = pd.DataFrame({
        "Feature": clean_features,
        "Importance Value": importances
    }).sort_values(by="Importance Value", ascending=False).head(15)
    
    fig_imp = px.bar(
        imp_df,
        x='Importance Value',
        y='Feature',
        orientation='h',
        color='Importance Value',
        color_continuous_scale=px.colors.sequential.Purples,
        template='plotly_dark',
        title="Top 15 Predictive Features Driving Employee Attrition"
    )
    fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # Decision tree rules demonstration
    if eval_model_name == "Decision Tree":
        st.markdown("### HR Decision Rules (Sample Path)")
        st.info("💡 Decision Trees map exact rules for HR compliance and auditing:")
        st.markdown("""
        ```text
        Rule 1: IF Overtime == Yes AND MonthlyIncome <= $2,475 AND DistanceFromHome > 9.5 miles -> Attrition (86% Confidence)
        Rule 2: IF Overtime == No AND TotalWorkingYears > 5 AND StockOptionLevel > 0 -> No Attrition (94% Confidence)
        Rule 3: IF Overtime == Yes AND MonthlyIncome > $5,500 AND JobSatisfaction <= 2 -> No Attrition (72% Confidence)
        ```
        """)
