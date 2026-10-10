import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Alzheimer's Disease Prediction",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================
@st.cache_resource
def load_artifacts():

    BASE_DIR = Path(__file__).resolve().parent

    model = joblib.load(
        BASE_DIR / "final_gradient_boosting.pkl"
    )

    preprocessor = joblib.load(
        BASE_DIR / "preprocessor.pkl"
    )

    return model, preprocessor


# Load artifacts
model, preprocessor = load_artifacts() 


# ============================================================
# HEADER
# ============================================================

st.title("🧠 Alzheimer's Disease Prediction")

st.markdown(
    """
    This application uses a trained **Gradient Boosting Classifier**
    to predict the diagnosis class based on the provided patient
    information.
    """
)

st.warning(
    "⚠️ This application is for educational and demonstration "
    "purposes only. It is not a medical diagnostic tool."
)


# ============================================================
# 1. DEMOGRAPHIC INFORMATION
# ============================================================

st.header("1. Demographic Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=60,
        max_value=90,
        value=75,
        step=1
    )


with col2:

    gender = st.selectbox(
        "Gender",
        options=[0, 1],
        format_func=lambda x:
            "Female" if x == 0 else "Male"
    )


with col3:

    ethnicity = st.selectbox(
        "Ethnicity",
        options=[0, 1, 2, 3]
    )


education_level = st.selectbox(
    "Education Level",
    options=[0, 1, 2, 3]
)


# ============================================================
# 2. LIFESTYLE INFORMATION
# ============================================================

st.header("2. Lifestyle Information")

col1, col2, col3 = st.columns(3)


with col1:

    bmi = st.number_input(
        "BMI",
        min_value=15.0,
        max_value=40.0,
        value=27.7,
        step=0.1
    )


with col2:

    smoking = st.selectbox(
        "Smoking",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


with col3:

    alcohol_consumption = st.number_input(
        "Alcohol Consumption",
        min_value=0.0,
        max_value=20.0,
        value=10.0,
        step=0.1
    )


col1, col2, col3 = st.columns(3)


with col1:

    physical_activity = st.number_input(
        "Physical Activity",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )


with col2:

    diet_quality = st.number_input(
        "Diet Quality",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )


with col3:

    sleep_quality = st.number_input(
        "Sleep Quality",
        min_value=4.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )


# ============================================================
# 3. MEDICAL HISTORY
# ============================================================

st.header("3. Medical History")

col1, col2, col3, col4 = st.columns(4)


with col1:

    family_history = st.selectbox(
        "Family History of Alzheimer's",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


with col2:

    cardiovascular = st.selectbox(
        "Cardiovascular Disease",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


with col3:

    diabetes = st.selectbox(
        "Diabetes",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


with col4:

    depression = st.selectbox(
        "Depression",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


col1, col2 = st.columns(2)


with col1:

    head_injury = st.selectbox(
        "Head Injury",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


with col2:

    hypertension = st.selectbox(
        "Hypertension",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


# ============================================================
# 4. COGNITIVE AND BEHAVIORAL SYMPTOMS
# ============================================================

st.header("4. Cognitive & Behavioral Symptoms")

col1, col2, col3, col4 = st.columns(4)


with col1:

    memory_complaints = st.selectbox(
        "Memory Complaints",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


with col2:

    behavioral_problems = st.selectbox(
        "Behavioral Problems",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


with col3:

    confusion = st.selectbox(
        "Confusion",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


with col4:

    disorientation = st.selectbox(
        "Disorientation",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


col1, col2, col3 = st.columns(3)


with col1:

    personality_changes = st.selectbox(
        "Personality Changes",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


with col2:

    difficulty_tasks = st.selectbox(
        "Difficulty Completing Tasks",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


with col3:

    forgetfulness = st.selectbox(
        "Forgetfulness",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


# ============================================================
# 5. BLOOD PRESSURE
# ============================================================

st.header("5. Blood Pressure")

col1, col2 = st.columns(2)


with col1:

    systolic_bp = st.number_input(
        "Systolic BP",
        min_value=90,
        max_value=179,
        value=134,
        step=1
    )


with col2:

    diastolic_bp = st.number_input(
        "Diastolic BP",
        min_value=60,
        max_value=119,
        value=90,
        step=1
    )


# ============================================================
# 6. CHOLESTEROL INFORMATION
# ============================================================

st.header("6. Cholesterol Information")

col1, col2 = st.columns(2)


with col1:

    cholesterol_total = st.number_input(
        "Total Cholesterol",
        min_value=150.0,
        max_value=300.0,
        value=225.0,
        step=0.1
    )


with col2:

    cholesterol_ldl = st.number_input(
        "LDL Cholesterol",
        min_value=50.0,
        max_value=200.0,
        value=124.0,
        step=0.1
    )


col1, col2 = st.columns(2)


with col1:

    cholesterol_hdl = st.number_input(
        "HDL Cholesterol",
        min_value=20.0,
        max_value=100.0,
        value=59.5,
        step=0.1
    )


with col2:

    cholesterol_triglycerides = st.number_input(
        "Triglycerides",
        min_value=50.0,
        max_value=400.0,
        value=228.0,
        step=0.1
    )


# ============================================================
# 7. COGNITIVE AND FUNCTIONAL ASSESSMENT
# ============================================================

st.header("7. Cognitive & Functional Assessment")

col1, col2, col3 = st.columns(3)


with col1:

    mmse = st.number_input(
        "MMSE",
        min_value=0.0,
        max_value=30.0,
        value=15.0,
        step=0.1
    )


with col2:

    functional_assessment = st.number_input(
        "Functional Assessment",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )


with col3:

    adl = st.number_input(
        "ADL",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Alzheimer's Risk",
    use_container_width=True
)


if predict_button:

    try:

        # ====================================================
        # FEATURE ENGINEERING
        # ====================================================

        cognitive_symptom_count = (
            memory_complaints
            + behavioral_problems
            + confusion
            + disorientation
            + personality_changes
            + difficulty_tasks
            + forgetfulness
        )


        cardiometabolic_risk_count = (
            smoking
            + cardiovascular
            + diabetes
            + hypertension
        )


        pulse_pressure = (
            systolic_bp
            - diastolic_bp
        )


        total_hdl_ratio = (
            cholesterol_total
            / cholesterol_hdl
        )


        # ====================================================
        # CREATE INPUT DATAFRAME
        # ====================================================

        input_data = pd.DataFrame({

            "Age": [age],

            "Gender": [gender],

            "Ethnicity": [ethnicity],

            "EducationLevel": [education_level],

            "BMI": [bmi],

            "Smoking": [smoking],

            "AlcoholConsumption": [
                alcohol_consumption
            ],

            "PhysicalActivity": [
                physical_activity
            ],

            "DietQuality": [
                diet_quality
            ],

            "SleepQuality": [
                sleep_quality
            ],

            "FamilyHistoryAlzheimers": [
                family_history
            ],

            "CardiovascularDisease": [
                cardiovascular
            ],

            "Diabetes": [
                diabetes
            ],

            "Depression": [
                depression
            ],

            "HeadInjury": [
                head_injury
            ],

            "Hypertension": [
                hypertension
            ],

            "SystolicBP": [
                systolic_bp
            ],

            "DiastolicBP": [
                diastolic_bp
            ],

            "CholesterolTotal": [
                cholesterol_total
            ],

            "CholesterolLDL": [
                cholesterol_ldl
            ],

            "CholesterolHDL": [
                cholesterol_hdl
            ],

            "CholesterolTriglycerides": [
                cholesterol_triglycerides
            ],

            "MMSE": [
                mmse
            ],

            "FunctionalAssessment": [
                functional_assessment
            ],

            "MemoryComplaints": [
                memory_complaints
            ],

            "BehavioralProblems": [
                behavioral_problems
            ],

            "ADL": [
                adl
            ],

            "Confusion": [
                confusion
            ],

            "Disorientation": [
                disorientation
            ],

            "PersonalityChanges": [
                personality_changes
            ],

            "DifficultyCompletingTasks": [
                difficulty_tasks
            ],

            "Forgetfulness": [
                forgetfulness
            ],

            "CognitiveSymptomCount": [
                cognitive_symptom_count
            ],

            "CardiometabolicRiskCount": [
                cardiometabolic_risk_count
            ],

            "PulsePressure": [
                pulse_pressure
            ],

            "Total_HDL_Ratio": [
                total_hdl_ratio
            ]
        })


        # ====================================================
        # ENSURE CORRECT COLUMN ORDER
        # ====================================================

        

        input_data = input_data[
            [
                "Age",
                "Gender",
                "Ethnicity",
                "EducationLevel",
                "BMI",
                "Smoking",
                "AlcoholConsumption",
                "PhysicalActivity",
                "DietQuality",
                "SleepQuality",
                "FamilyHistoryAlzheimers",
                "CardiovascularDisease",
                "Diabetes",
                "Depression",
                "HeadInjury",
                "Hypertension",
                "SystolicBP",
                "DiastolicBP",
                "CholesterolTotal",
                "CholesterolLDL",
                "CholesterolHDL",
                "CholesterolTriglycerides",
                "MMSE",
                "FunctionalAssessment",
                "MemoryComplaints",
                "BehavioralProblems",
                "ADL",
                "Confusion",
                "Disorientation",
                "PersonalityChanges",
                "DifficultyCompletingTasks",
                "Forgetfulness",
                "CognitiveSymptomCount",
                "CardiometabolicRiskCount",
                "PulsePressure",
                "Total_HDL_Ratio"
            ]
        ]


        # ====================================================
        # PREPROCESS INPUT
        # ====================================================

        processed_input = preprocessor.transform(
            input_data
        )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = model.predict(
            processed_input
        )[0]


        probabilities = model.predict_proba(
            processed_input
        )[0]


        negative_probability = probabilities[0]

        positive_probability = probabilities[1]


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.divider()

        st.subheader("Prediction Result")


        if prediction == 1:

            st.error(
                "🔴 Model Prediction: Alzheimer's"
            )

        else:

            st.success(
                "🟢 Model Prediction: No Alzheimer's"
            )


        # ====================================================
        # PROBABILITY DISPLAY
        # ====================================================

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "No Alzheimer's Probability",
                f"{negative_probability * 100:.2f}%"
            )


        with col2:

            st.metric(
                "Alzheimer's Probability",
                f"{positive_probability * 100:.2f}%"
            )


        # ====================================================
        # ENGINEERED FEATURE INFORMATION
        # ====================================================

        with st.expander(
            "View Engineered Features"
        ):

            engineered_features = pd.DataFrame({

                "Feature": [
                    "CognitiveSymptomCount",
                    "CardiometabolicRiskCount",
                    "PulsePressure",
                    "Total_HDL_Ratio"
                ],

                "Value": [
                    cognitive_symptom_count,
                    cardiometabolic_risk_count,
                    pulse_pressure,
                    total_hdl_ratio
                ]
            })

            st.dataframe(
                engineered_features,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            "An error occurred while generating "
            "the prediction."
        )

        st.exception(e)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader("📊 Model Information")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Model",
        "Gradient Boosting"
    )


with col2:

    st.metric(
        "Accuracy",
        "95.58%"
    )


with col3:

    st.metric(
        "F1 Score",
        "93.69%"
    )


with col4:

    st.metric(
        "ROC-AUC",
        "94.94%"
    )


st.caption(
    "Model performance metrics are based on the held-out "
    "test dataset used during model evaluation."
)

st.caption(
    "This prediction should not be interpreted as a medical diagnosis."
)