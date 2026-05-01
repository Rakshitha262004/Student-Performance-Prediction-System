import pandas as pd
import numpy as np
import os

def generate_student_data(n=500):
    np.random.seed(42)

    df = pd.DataFrame({
        'student_id': range(1, n+1),
        'study_hours_per_day': np.random.uniform(1, 10, n),
        'attendance_percent': np.random.uniform(50, 100, n),
        'previous_score': np.random.uniform(40, 100, n),
        'sleep_hours': np.random.uniform(4, 9, n),
        'extracurricular': np.random.choice([0, 1], n),
        'parent_education': np.random.choice(['School', 'Graduate', 'Postgraduate'], n),
        'internet_access': np.random.choice([0, 1], n),
        'tutoring': np.random.choice([0, 1], n),
        'family_income': np.random.choice(['Low', 'Medium', 'High'], n),
        'motivation_score': np.random.randint(1, 10, n),
        'absences': np.random.randint(0, 20, n)
    })

    score = (
        df['study_hours_per_day'] * 5 +
        df['attendance_percent'] * 0.2 +
        df['previous_score'] * 0.4 +
        df['sleep_hours'] * 2 +
        df['motivation_score'] * 3 -
        df['absences'] * 1.5 +
        np.random.normal(0, 5, n)
    )

    df['final_score'] = score

    df['performance'] = pd.cut(
        df['final_score'],
        bins=[-np.inf, 50, 75, np.inf],
        labels=['Low', 'Medium', 'High']
    )

    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/student_data.csv', index=False)

    print("✅ Dataset generated")

if __name__ == "__main__":
    generate_student_data()