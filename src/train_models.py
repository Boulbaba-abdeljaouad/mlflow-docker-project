import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# -----------------------------
# Config
# -----------------------------
DATA_PATH = "data/UCI_Credit_Card.csv"
EXPERIMENT_NAME = "Credit Default Classification"
TARGET_COL = "default.payment.next.month"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Drop ID column
X = df.drop(columns=["ID", TARGET_COL])
y = df[TARGET_COL]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Models to compare
# -----------------------------
models = [
    (
        "LogisticRegression_lbfgs",
        {"max_iter": 1000, "solver": "lbfgs", "C": 1.0},
        LogisticRegression()
    ),
    (
        "RandomForest_100",
        {"n_estimators": 100, "max_depth": 7, "random_state": 42},
        RandomForestClassifier()
    ),
    (
        "GradientBoosting_100",
        {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
        GradientBoostingClassifier()
    ),
    (
        "RandomForest_200",
        {"n_estimators": 200, "max_depth": 10, "random_state": 42},
        RandomForestClassifier()
    ),
    (
        "LogisticRegression_l2_stronger",
        {"max_iter": 1000, "solver": "lbfgs", "C": 0.5},
        LogisticRegression()
    ),
]

# -----------------------------
# MLflow setup
# -----------------------------

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Credit Default Classification")
for model_name, params, model in models:
    with mlflow.start_run(run_name=model_name):
        model.set_params(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log params
        mlflow.log_params(params)
        mlflow.log_param("model_name", model_name)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Finished {model_name}")
        print({
            "accuracy": round(accuracy, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1_score": round(f1, 4)
        })