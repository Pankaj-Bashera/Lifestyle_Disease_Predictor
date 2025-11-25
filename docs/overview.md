# Lifestyle Disease Predictor Documentation

## 1. Overview

The Lifestyle Disease Predictor is a machine-learning project designed to estimate the risk of lifestyle-associated diseases based on personal health metrics and daily habits. It functions as a decision-support system, helping users understand potential health risks and encouraging preventive lifestyle changes.

---

## 2. Problem Statement

Lifestyle diseases such as diabetes, hypertension, obesity, and cardiovascular disorders often develop due to long-term behavioral patterns. This project aims to:

* Analyze structured health and lifestyle data.
* Train ML models to classify individuals into disease risk categories.
* Enable predictions for new user inputs.
* Provide a foundation for future deployment as a wellness assistant or personal-health dashboard.

This tool offers risk estimation, not clinical diagnosis.

---

## 3. Dataset Description

The dataset is a tabular CSV where each row represents one individual. Columns include health metrics, lifestyle habits, and the final risk classification.

### 3.1 Columns

* **Age**: Integer representing age in years.
* **Gender**: Male/Female (may include inconsistencies like casing or typos).
* **BMI**: Body Mass Index.
* **Smoking**: Yes/No.
* **Alcohol**: Yes/No.
* **ExerciseHours**: Average daily exercise duration.
* **SleepHours**: Average daily sleep duration.
* **DietScore**: Integer rating of diet quality.
* **BloodPressure**: Systolic blood pressure.
* **BloodSugar**: Blood sugar level.
* **Cholesterol**: Cholesterol measurement.
* **DiseaseRisk**: Target label (Low/High).

### 3.2 Unprocessed Data Handling

The project supports raw, messy datasets containing:

* Missing values.
* Random spaces or inconsistent formatting.
* Casing inconsistencies.
* Minor typographical errors.
* Extra delimiters.

The preprocessing pipeline cleans and normalizes these issues.

---

## 4. Repository Structure

```
Lifestyle_Disease_Predictor/
│
├── data/                     # Raw or cleaned CSV datasets
├── models/                   # Saved model files (.pkl/.joblib)
├── notebooks/                # Jupyter notebooks for EDA & model training
├── requirements.txt          # Project dependencies
└── README.md                 # Primary project overview
```

A recommended future structure for scripts:

```
src/
├── preprocess.py
├── model.py
└── predict.py
main.py
```

---

## 5. Installation

### 5.1 Prerequisites

* Python 3.8+
* Git
* Virtual environment (optional but recommended)

### 5.2 Clone the Repository

```
git clone https://github.com/Pankaj-Bashera/Lifestyle_Disease_Predictor.git
cd Lifestyle_Disease_Predictor
```

### 5.3 Install Dependencies

```
pip install -r requirements.txt
```

---

## 6. Usage

### 6.1 Running via Jupyter Notebook

```
jupyter notebook
```

Open notebooks in `/notebooks` to perform:

* Data loading
* Cleaning and preprocessing
* Model training
* Evaluation and visualization

### 6.2 Typical Training Workflow

1. **Load the dataset** using pandas.
2. **Separate features and target** (`DiseaseRisk`).
3. **Apply preprocessing**: handle missing values, normalize text, encode categorical variables, scale numerics if needed.
4. **Split** into training and testing sets.
5. **Train models** such as Logistic Regression or Random Forest.
6. **Evaluate** using accuracy, precision, recall, and F1-score.
7. **Save the best model** into `/models` using joblib.

### 6.3 Predicting on New Data

Load the trained model and preprocess inputs identically before prediction.

---

## 7. Machine Learning Pipeline

1. **Data Collection**: ingest CSV files.
2. **Preprocessing**: handle noise, inconsistencies, missing values.
3. **Feature Engineering**: derive additional insights (optional).
4. **Model Training**: Logistic Regression, Random Forest, or others.
5. **Evaluation**: confusion matrix, precision/recall, etc.
6. **Persistence**: save model and preprocessors.

---

## 8. Interpreting Predictions

The model outputs categories such as:

* **Low Risk**: indicators of lower lifestyle-disease likelihood.
* **High Risk**: elevated concern requiring behavioral changes.

Probabilistic risk outputs may be added later.

---

## 9. Future Extensions

* Web dashboard using Streamlit or FastAPI.
* SHAP or LIME for explainability.
* Larger and more diverse datasets.
* Time-series user health tracking.
* Compare multiple ML models with visualization.

---

## 10. Limitations

* Not medically certified.
* Depends on data quality.
* Susceptible to bias present in training data.
* Should not be used as a substitute for professional medical advice.

---

## 11. Contributing

1. Fork the repository.
2. Create a feature branch.
3. Commit changes with clear messages.
4. Open a pull request.

---

## 12. Authors

* **Pankaj Bashera** – B.Tech CSE (AIML), LPU

  * GitHub: [https://github.com/Pankaj-Bashera](https://github.com/Pankaj-Bashera)
  * LinkedIn: [https://www.linkedin.com/in/pankajb1](https://www.linkedin.com/in/pankajb1)

* **Jatin Kumar** – B.Tech CSE (AIML), LPU

For bugs or suggestions, open an issue on GitHub.

---

