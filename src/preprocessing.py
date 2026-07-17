import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_dataset(df):

    # --------------------------
    # Remove duplicate rows
    # --------------------------

    df = df.drop_duplicates()

    # --------------------------
    # Fill missing values
    # --------------------------

    for column in df.columns:

        if df[column].dtype == "object":

            df[column] = df[column].fillna(df[column].mode()[0])

        else:

            df[column] = df[column].fillna(df[column].median())

    # --------------------------
    # Encode categorical columns
    # --------------------------

    encoder = LabelEncoder()

    categorical_columns = [
        "gender",
        "smoking_history"
    ]

    for column in categorical_columns:

        df[column] = encoder.fit_transform(df[column])

    return df