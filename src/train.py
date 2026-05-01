import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

from preprocess import load_data, preprocess, split_features_target
from features import engineer_features

def train_model():
    os.makedirs('models', exist_ok=True)

    df = load_data()
    df, encoders = preprocess(df)
    df = engineer_features(df)

    X, y = split_features_target(df)

    # Remove NaN
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    le_target = LabelEncoder()
    y_enc = le_target.fit_transform(y)
    joblib.dump(le_target, 'models/target_encoder.pkl')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    print(classification_report(
        y_test,
        y_pred,
        target_names=[str(c) for c in le_target.classes_],
        zero_division=0
    ))

    joblib.dump(model, 'models/rf_model.pkl')
    print("✅ Model saved")

    return model

if __name__ == "__main__":
    train_model()