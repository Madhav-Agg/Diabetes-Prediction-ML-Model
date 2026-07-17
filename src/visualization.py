import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)

# Create graphs folder if it doesn't exist
os.makedirs("graphs", exist_ok=True)


# ---------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------

def correlation_heatmap(df):

    plt.figure(figsize=(10,8))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig("graphs/correlation_heatmap.png")

    plt.close()


# ---------------------------------------------------
# Diabetes Class Distribution
# ---------------------------------------------------

def class_distribution(df):

    plt.figure(figsize=(6,5))

    sns.countplot(
        x="diabetes",
        data=df
    )

    plt.title("Diabetes Distribution")

    plt.tight_layout()

    plt.savefig("graphs/class_distribution.png")

    plt.close()


# ---------------------------------------------------
# Age Distribution
# ---------------------------------------------------

def age_distribution(df):

    plt.figure(figsize=(8,5))

    sns.histplot(
        df["age"],
        bins=30,
        kde=True
    )

    plt.title("Age Distribution")

    plt.tight_layout()

    plt.savefig("graphs/age_distribution.png")

    plt.close()


# ---------------------------------------------------
# BMI Distribution
# ---------------------------------------------------

def bmi_distribution(df):

    plt.figure(figsize=(8,5))

    sns.histplot(
        df["bmi"],
        bins=30,
        kde=True
    )

    plt.title("BMI Distribution")

    plt.tight_layout()

    plt.savefig("graphs/bmi_distribution.png")

    plt.close()


# ---------------------------------------------------
# Blood Glucose Distribution
# ---------------------------------------------------

def glucose_distribution(df):

    plt.figure(figsize=(8,5))

    sns.histplot(
        df["blood_glucose_level"],
        bins=30,
        kde=True
    )

    plt.title("Blood Glucose Distribution")

    plt.tight_layout()

    plt.savefig("graphs/glucose_distribution.png")

    plt.close()


# ---------------------------------------------------
# Boxplots
# ---------------------------------------------------

def boxplots(df):

    plt.figure(figsize=(14,6))

    sns.boxplot(data=df.select_dtypes(include="number"))

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig("graphs/boxplots.png")

    plt.close()


# ---------------------------------------------------
# Model Accuracy Comparison
# ---------------------------------------------------

def model_accuracy_graph(results):

    plt.figure(figsize=(10,6))

    bars = plt.bar(
        results.keys(),
        results.values()
    )

    plt.title("Model Accuracy Comparison")

    plt.ylabel("Accuracy")

    plt.ylim(0.80,1.00)

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.003,
            f"{height*100:.2f}%",
            ha="center"
        )

    plt.tight_layout()

    plt.savefig("graphs/model_accuracy.png")

    plt.close()


# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------

def confusion_matrix_graph(y_test, prediction):

    cm = confusion_matrix(
        y_test,
        prediction
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig("graphs/confusion_matrix.png")

    plt.close()


# ---------------------------------------------------
# ROC Curve
# ---------------------------------------------------

def roc_curve_graph(y_test, probabilities):

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(6,6))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.3f}"
    )

    plt.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig("graphs/roc_curve.png")

    plt.close()