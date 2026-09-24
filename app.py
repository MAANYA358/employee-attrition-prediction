import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

# Load model
model = joblib.load("models/attrition_model.pkl")

st.title("📊 Employee Attrition Prediction")
st.write("Predict whether an employee is likely to leave the organization.")

st.divider()

st.subheader("Employee Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 70, 30)
    monthly_income = st.number_input("Monthly Income", 1000, 50000, 5000)
    distance = st.number_input("Distance From Home", 1, 100, 5)
    years_company = st.number_input("Years At Company", 0, 50, 5)

with col2:
    job_satisfaction = st.selectbox(
        "Job Satisfaction", [1, 2, 3, 4], index=2
    )
    environment = st.selectbox(
        "Environment Satisfaction", [1, 2, 3, 4], index=2
    )
    job_level = st.selectbox(
        "Job Level", [1, 2, 3, 4, 5], index=0
    )
    overtime = st.selectbox(
        "Overtime", ["Yes", "No"]
    )

with col3:
    business_travel = st.selectbox(
        "Business Travel",
        ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
    )

    department = st.selectbox(
        "Department",
        ["Sales", "Research & Development", "Human Resources"]
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

education = st.selectbox("Education", [1, 2, 3, 4, 5])
total_working_years = st.number_input(
    "Total Working Years", 0, 50, 5
)

if st.button("🔍 Predict Attrition", use_container_width=True):

    input_data = pd.DataFrame([{
        "Age": age,
        "DailyRate": 500,
        "BusinessTravel": business_travel,
        "Department": department,
        "DistanceFromHome": distance,
        "Education": education,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": environment,
        "Gender": "Male",
        "HourlyRate": 60,
        "JobInvolvement": 3,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": 2,
        "OverTime": overtime,
        "PercentSalaryHike": 15,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 3,
        "YearsAtCompany": years_company,
        "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    if prediction == 1:
        st.error("⚠️ HIGH ATTRITION RISK")
    else:
        st.success("✅ LOW ATTRITION RISK")

    st.metric(
        "Attrition Probability",
        f"{probability * 100:.2f}%"
    )