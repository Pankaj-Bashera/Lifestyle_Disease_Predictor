import streamlit as st
import joblib
import numpy as np

# Import custom models
from algo.logistic_regressor import Logistic_Regressor
from algo.knn import Classifier


st.set_page_config(page_title="Lifestyle Disease Risk Predictor", layout="centered")

st.title("Lifestyle Disease Risk Predictor")
st.subheader("Select Algorithm & Enter Health Data")

# Load models
@st.cache_resource
def load_models():
    return {
        "Logistic_Regressor": joblib.load("models/log_reg.pkl"),
        "Classifier": joblib.load("models/knn.pkl"),
        "Naive Bayes": joblib.load("models/naive_bayes.pkl"),   # sklearn → loads fine
    }

models = load_models()
model_name = st.selectbox("Choose ML Model", list(models.keys()))
model = models[model_name]

# --- INPUT FIELDS (all 11 features) ---
st.write("### Enter Patient Details")

Age = st.number_input("Age", 1, 120)
Gender = st.selectbox("Gender", [0, 1])  # assuming 0/1 encoded
BMI = st.number_input("BMI", 10.0, 60.0)
Smoking = st.selectbox("Smoking", [0, 1])
Alcohol = st.selectbox("Alcohol Consumption", [0, 1])
ExerciseHours = st.number_input("Exercise Hours Per Day", 0.0, 5.0)
SleepHours = st.number_input("Sleep Hours Per Day", 0.0, 12.0)
DietScore = st.number_input("Diet Score (0–10)", 0, 10)
BloodPressure = st.number_input("Blood Pressure", 50, 200)
BloodSugar = st.number_input("Blood Sugar", 50, 300)
Cholesterol = st.number_input("Cholesterol", 100, 400)

# --- PREDICT ---
if st.button("Predict Risk"):
    features = np.array([[
        Age, Gender, BMI, Smoking, Alcohol,
        ExerciseHours, SleepHours, DietScore,
        BloodPressure, BloodSugar, Cholesterol
    ]])

    pred = model.predict(features)[0]
    st.success(f"Predicted Risk: {pred}")
