import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_data(path='data/raw/student_data.csv'):
    return pd.read_csv(path)

def preprocess(df):
    df = df.copy()

    df.drop('student_id', axis=1, inplace=True, errors='ignore')

    cat_cols = ['parent_education', 'family_income']
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    os.makedirs('models', exist_ok=True)
    joblib.dump(encoders, 'models/encoders.pkl')

    return df, encoders

def split_features_target(df, target='performance'):
    X = df.drop([target, 'final_score'], axis=1, errors='ignore')
    y = df[target]
    return X, y