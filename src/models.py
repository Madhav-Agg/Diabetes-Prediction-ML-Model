import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier




def train_models(X_train, X_test, y_train, y_test):
    models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),

    
        "XGBoost":
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss"
            )

}

    results = {}

    best_model = None
    best_prediction = None

    best_accuracy = 0

    print("\n" + "="*70)
    print("Training Models")
    print("="*70)

    for name, model in models.items():

        print(f"\n{name}")

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        cv = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5
        ).mean()

        print(f"Accuracy        : {accuracy*100:.2f}%")
        print(f"Cross Validation: {cv*100:.2f}%")

        results[name] = accuracy

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_model = model
            best_prediction = prediction

    print("\n" + "="*70)
    print("Training Completed")
    print("="*70)

    print("Best Model :", best_model.__class__.__name__)
    print("Accuracy   :", round(best_accuracy*100,2), "%")

    result_df = pd.DataFrame({

        "Model": results.keys(),
        "Accuracy": results.values()

    })

    result_df.to_csv(
        "graphs/model_results.csv",
        index=False
    )

    return (
        best_model,
        best_prediction,
        results
    )