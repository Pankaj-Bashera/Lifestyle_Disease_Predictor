import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Disease Risk Predictor",
    page_icon="jq",
    layout="wide"
)

# --- CSS for better styling ---
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .risk-high {
        padding: 20px;
        background-color: #ffe6e6;
        border-radius: 10px;
        color: #cc0000;
        text-align: center;
    }
    .risk-low {
        padding: 20px;
        background-color: #e6fffa;
        border-radius: 10px;
        color: #006644;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar: Model Management ---
st.sidebar.header("Model Configuration")
model_source = st.sidebar.radio("Prediction Source", ["Demo Mode (Random)", "Load .pkl File"])

model = None

# Logic to load real .pkl model
if model_source == "Load .pkl File":
    uploaded_file = st.sidebar.file_uploader("Upload your trained model (.pkl)", type=['pkl'])
    if uploaded_file is not None:
        try:
            model = joblib.load(uploaded_file)
            st.sidebar.success("Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
    else:
        st.sidebar.warning("Please upload a .pkl file to proceed with real predictions.")

# --- Main App Interface ---
st.title("❤️ Disease Risk Prediction System")
st.markdown("Enter the patient's health parameters below to assess disease risk.")

with st.form("risk_assessment_form"):
    st.subheader("1. Patient Demographics & Body Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    with col2:
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=22.5, format="%.1f")
        # Visual Helper for BMI
        if bmi < 18.5:
            st.caption("Status: Underweight")
        elif 18.5 <= bmi < 25:
            st.caption("Status: Healthy")
        elif 25 <= bmi < 30:
            st.caption("Status: Overweight")
        else:
            st.caption("Status: Obese")
            
    with col3:
        # Assuming Systolic for a single BP field, or map to average
        bp = st.number_input("Blood Pressure (Systolic)", min_value=50, max_value=250, value=120)

    st.markdown("---")
    st.subheader("2. Lifestyle Factors")
    col4, col5, col6 = st.columns(3)

    with col4:
        smoking = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker"])
        alcohol = st.selectbox("Alcohol Consumption", ["None", "Moderate", "High"])

    with col5:
        exercise = st.slider("Exercise Hours (per week)", 0, 20, 3)
        sleep = st.slider("Sleep Hours (per night)", 0, 12, 7)

    with col6:
        diet_score = st.slider("Diet Score (1=Poor, 10=Healthy)", 1, 10, 5)

    st.markdown("---")
    st.subheader("3. Clinical Metrics")
    col7, col8 = st.columns(2)

    with col7:
        sugar = st.number_input("Blood Sugar (mg/dL)", min_value=50, max_value=400, value=90)
    
    with col8:
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)

    # Submit Button
    submitted = st.form_submit_button("Analyze Risk")

# --- Prediction Logic ---
if submitted:
    # 1. Preprocess Input
    # Convert categorical text to numbers (Example logic - adjust based on your specific model training)
    gender_map = {"Male": 1, "Female": 0, "Other": 2}
    smoking_map = {"Non-Smoker": 0, "Smoker": 1}
    alcohol_map = {"None": 0, "Moderate": 1, "High": 2}

    # Create a DataFrame matching the columns used during training
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_map[gender]],
        'BMI': [bmi],
        'Smoking': [smoking_map[smoking]],
        'Alcohol': [alcohol_map[alcohol]],
        'ExerciseHours': [exercise],
        'SleepHours': [sleep],
        'DietScore': [diet_score],
        'BloodPressure': [bp],
        'BloodSugar': [sugar],
        'Cholesterol': [cholesterol]
    })

    st.info("Input Data Processing Complete")
    
    # Optional: Show input data for debugging
    with st.expander("View Input Dataframe"):
        st.dataframe(input_data)

    # 2. Make Prediction
    try:
        if model_source == "Demo Mode (Random)":
            # --- MOCK PREDICTION FOR DEMO ---
            import time
            with st.spinner('Calculating risk factors...'):
                time.sleep(1) # Simulate computation
                
                # Simple heuristic for demo purposes
                risk_score = 0
                if bmi > 30: risk_score += 1
                if smoking == "Smoker": risk_score += 1
                if age > 60: risk_score += 1
                if sugar > 140: risk_score += 1
                
                prediction = 1 if risk_score >= 2 else 0
                probability = 0.85 if prediction == 1 else 0.15

        elif model_source == "Load .pkl File" and model is not None:
            # --- REAL PREDICTION ---
            # Ensure input_data columns match exactly what the model expects
            prediction = model.predict(input_data)[0]
            
            # Check if model supports probability (some classifiers do, regressors don't)
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(input_data)[0][1]
            else:
                probability = None
        else:
            st.error("Model not loaded. Please upload a .pkl file in the sidebar.")
            st.stop()

        # 3. Display Results
        st.markdown("### Analysis Result")
        
        if prediction == 1:
            st.markdown(
                f"""
                <div class="risk-high">
                    <h2>⚠️ High Disease Risk Detected</h2>
                    {f'<p>Probability: {probability:.1%}</p>' if probability else ''}
                    The model suggests a higher likelihood of disease based on the provided parameters.
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.warning("Recommendation: Please consult a healthcare professional for a detailed checkup.")
        else:
            st.markdown(
                f"""
                <div class="risk-low">
                    <h2>✅ Low Disease Risk</h2>
                    {f'<p>Probability: {probability:.1%}</p>' if probability else ''}
                    The patient appears to be within healthy parameters based on this model.
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.balloons()

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.markdown("Ensure that the input features in your .pkl model match the order and names of the inputs above.")

# Footer
st.markdown("---")
st.caption("Note: This is a medical decision support tool prototype. Results should be verified by a doctor.")