def engineer_features(df):
    df = df.copy()

    df['study_efficiency'] = df['study_hours_per_day'] / (df['sleep_hours'] + 1)

    df['engagement_score'] = (
        df['extracurricular'] * 2 +
        df['motivation_score'] +
        df['tutoring'] * 3
    )

    df['attendance_health'] = df['attendance_percent'] - df['absences'] * 0.5

    income_map = {0: 1, 1: 2, 2: 3}
    df['resource_advantage'] = df['internet_access'] + df['family_income'].map(income_map)

    return df