# 🩺 Lifestyle Disease Predictor

A Machine Learning model that predicts the likelihood of lifestyle-related diseases (like diabetes, heart disease, or obesity) based on health and lifestyle inputs such as diet, exercise, sleep, and stress levels.

---

## 📘 Overview

Lifestyle diseases are on the rise due to poor daily habits. This project aims to provide an intelligent tool that predicts potential disease risks early using machine learning, helping individuals take preventive actions.

---

## 🚀 Features

* Predicts the probability of lifestyle diseases
* Interactive input system (manual or CSV upload)
* Data preprocessing and feature engineering pipeline
* Trained ML models (Logistic Regression, Random Forest, etc.)
* Easy to integrate with a web or mobile interface

---

## 🧠 Machine Learning Workflow

1. **Data Collection:** Health and lifestyle dataset (public or custom)
2. **Data Cleaning & Preprocessing:** Handling missing values, encoding categorical data
3. **Feature Selection:** Identifying key health and lifestyle factors
4. **Model Training:** Algorithms like Logistic Regression, Random Forest, or XGBoost
5. **Evaluation:** Accuracy, Precision, Recall, F1-score, ROC curve

---

## 📂 Project Structure

```
lifestyle-disease-predictor/
│
├── data/                  # Datasets (not included in repo)
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code for preprocessing and modeling
│   ├── preprocess.py
│   ├── model.py
│   └── predict.py
├── models/                # Saved trained models
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── main.py                # Entry point script
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Pankaj-Bashera/lifestyle-disease-predictor.git
cd lifestyle-disease-predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧬 Usage

To make predictions:

```bash
python main.py
```

Or open the Jupyter notebook:

```bash
jupyter notebook notebooks/model_training.ipynb
```

---

## 📊 Example Input

| Feature           | Example Value |
| ----------------- | ------------- |
| Age               | 35            |
| BMI               | 27.3          |
| Exercise per week | 3 days        |
| Smoking Habit     | No            |
| Sleep Hours       | 6             |

---

## 📈 Results

* **Accuracy:** 88%
* **Precision:** 0.86
* **Recall:** 0.84
* **F1 Score:** 0.85

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn
* **Tools:** Jupyter Notebook, Git, VS Code

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repo, make improvements, and submit a pull request.

---

## 🧑‍💻 Authors

**Pankaj Bashera**
📍 B.Tech CSE (AIML) @ Lovely Professional University
🔗 [GitHub](https://github.com/Pankaj-Bashera) | [LinkedIn](https://www.linkedin.com/in/pankajb1)

**Jatin**
📍 B.Tech CSE (AIML) @ Lovely Professional University
🔗 [GitHub](https://github.com/Jatinkumar2519) | [LinkedIn](https://www.linkedin.com/in/jatinturk)


---

