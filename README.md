# 🎓 Student Performance Prediction System

A Machine Learning project that predicts student academic performance (Low / Medium / High) using behavioral, academic, and socio-economic factors.

---

## 🚀 Overview

This project builds an **end-to-end ML pipeline** to analyze student data and predict performance. It helps identify at-risk students and supports data-driven educational decisions.

---

## 🎯 Key Features

- 📊 Predict student performance (Low / Medium / High)
- 🧠 Feature engineering for better accuracy
- 🌳 Random Forest model with class balancing
- ⚡ Modular pipeline (data → preprocessing → model → prediction)
- 🌐 Interactive web app using Streamlit

---

## 🧠 ML Approach

- **Problem Type:** Classification  
- **Model Used:** Random Forest Classifier  
- **Techniques:**
  - Label Encoding
  - Feature Engineering
  - Train/Test Split
  - Model Evaluation

---

## 📊 Features Used

- Study Hours  
- Attendance  
- Previous Score  
- Sleep Hours  
- Extracurricular Activities  
- Parent Education  
- Internet Access  
- Tutoring  
- Family Income  
- Motivation Score  
- Absences  

---

## 🔮 Engineered Features

- Study Efficiency  
- Engagement Score  
- Attendance Health  
- Resource Advantage  

---

## 🧪 Dataset

- Synthetic dataset generated using Python
- Simulates realistic student behavior

📁 `data/raw/student_data.csv`

---

## ⚙️ Installation

```bash
pip install pandas numpy scikit-learn streamlit joblib
▶️ Run the Project
Train model
python main.py
Run web app
streamlit run app.py
💻 Web App

Interactive UI to input student data and get instant predictions.

📈 Model Performance
Accuracy: ~65–70%
Balanced using class weights
Evaluated using classification metrics
📂 Project Structure
src/
 ├── generate_data.py
 ├── preprocess.py
 ├── features.py
 ├── train.py
 └── predict.py
🎯 Highlights
End-to-end ML pipeline
Real-world feature engineering
Clean modular code
Streamlit deployment
👩‍💻 Author

Rakshitha A S