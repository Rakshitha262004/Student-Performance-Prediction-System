import streamlit as st
import pandas as pd
import joblib
import sys

sys.path.insert(0, 'src')
from features import engineer_features

model = joblib.load('models/rf_model.pkl')
encoders = joblib.load('models/encoders.pkl')
le_target = joblib.load('models/target_encoder.pkl')

st.title("🎓 Student Performance Predictor")

study = st.slider("Study Hours", 1, 10, 5)
attendance = st.slider("Attendance", 50, 100, 80)
prev = st.slider("Previous Score", 40, 100, 70)
sleep = st.slider("Sleep", 4, 9, 7)

parent = st.selectbox("Parent Education", ["School", "Graduate", "Postgraduate"])
income = st.selectbox("Family Income", ["Low", "Medium", "High"])

extra = st.selectbox("Extracurricular", [0, 1])
internet = st.selectbox("Internet", [0, 1])
tutor = st.selectbox("Tutoring", [0, 1])
motivation = st.slider("Motivation", 1, 10, 7)
absences = st.slider("Absences", 0, 20, 5)

if st.button("Predict"):
    data = pd.DataFrame([{
        'study_hours_per_day': study,
        'attendance_percent': attendance,
        'previous_score': prev,
        'sleep_hours': sleep,
        'extracurricular': extra,
        'parent_education': parent,
        'internet_access': internet,
        'tutoring': tutor,
        'family_income': income,
        'motivation_score': motivation,
        'absences': absences
    }])

    for col, le in encoders.items():
        data[col] = le.transform(data[col])

    data = engineer_features(data)

    pred = model.predict(data)[0]
    label = le_target.inverse_transform([pred])[0]

    st.success(f"Prediction: {label}")