# ============================================
# Diabetes Prediction System
# train.py
# Part 1
# ============================================

import os
import joblib
import warnings

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.preprocessing import clean_dataset

from src.visualization import (
    correlation_heatmap,
    class_distribution,
    age_distribution,
    bmi_distribution,
    glucose_distribution,
    boxplots,
    model_accuracy_graph,
    confusion_matrix_graph,
    roc_curve_graph
)

from src.models import train_models

warnings.filterwarnings("ignore")

# ============================================
# Create folders
# ============================================

os.makedirs("graphs", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ============================================
# Load Dataset
# ============================================

print("="*70)
print("Loading Dataset")
print("="*70)

df = pd.read_csv("dataset/diabetes_prediction_dataset.csv")

print(df.head())

print("\nDataset Shape :", df.shape)

print("\nColumns")

print(df.columns)

print("\nMissing Values")

print(df.isnull().sum())

# ============================================
# Cleaning Dataset
# ============================================

print("\n"+"="*70)
print("Cleaning Dataset")
print("="*70)

df = clean_dataset(df)

print("Cleaning Completed!")

# ============================================
# Preparing Features
# ============================================

X = df.drop("diabetes", axis=1)

y = df["diabetes"]

print("\nFeatures :", X.shape)

print("Target :", y.shape)


# =====================================================
# Generate Graphs
# =====================================================

print("\nGenerating Graphs...")

correlation_heatmap(df)
class_distribution(df)
age_distribution(df)
bmi_distribution(df)
glucose_distribution(df)
boxplots(df)

print("Graphs Generated Successfully!")

# =====================================================
# Train Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples :", len(X_test))

# =====================================================
# Feature Scaling
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed!")

# =====================================================
# Train Machine Learning Models
# =====================================================

best_model, prediction, results = train_models(
    X_train,
    X_test,
    y_train,
    y_test
)

# =====================================================
# Generate Remaining Graphs
# =====================================================

print("\nGenerating Model Graphs...")

model_accuracy_graph(results)

if hasattr(best_model, "predict_proba"):

    probabilities = best_model.predict_proba(X_test)[:,1]

    roc_curve_graph(
        y_test,
        probabilities
    )

confusion_matrix_graph(
    y_test,
    prediction
)

print("Graphs Generated!")

# =====================================================
# Save Model
# =====================================================

joblib.dump(
    best_model,
    "model/diabetes_model.pkl"
)

joblib.dump(
    scaler,
    "model/scaler.pkl"
)

print("\nModel Saved Successfully!")