import pandas as pd
import joblib
from features import engineer_features

def predict_student(student_dict):
    model = joblib.load('models/rf_model.pkl')
    encoders = joblib.load('models/encoders.pkl')
    le_target = joblib.load('models/target_encoder.pkl')

    df = pd.DataFrame([student_dict])

    for col, le in encoders.items():
        df[col] = le.transform(df[col])

    df = engineer_features(df)

    pred = model.predict(df)[0]
    probs = model.predict_proba(df)[0]

    label = le_target.inverse_transform([pred])[0]

    return {
        "prediction": label,
        "confidence": max(probs)
    }