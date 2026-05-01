import sys
sys.path.insert(0, 'src')

from src.generate_data import generate_student_data
from train import train_model
from predict import predict_student

if __name__ == "__main__":
    print("🚀 Running Project")

    generate_student_data()
    train_model()

    sample = {
        'study_hours_per_day': 7,
        'attendance_percent': 90,
        'previous_score': 80,
        'sleep_hours': 7,
        'extracurricular': 1,
        'parent_education': 'Graduate',
        'internet_access': 1,
        'tutoring': 1,
        'family_income': 'High',
        'motivation_score': 9,
        'absences': 2
    }

    result = predict_student(sample)
    print(result)