import mlflow
import joblib
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "CreditDefaultModel"
MODEL_ALIAS = "Champion"
EXPORT_PATH = "model.pkl"

# -----------------------------
# Setup
# -----------------------------
mlflow.set_tracking_uri(TRACKING_URI)

# Load champion model from registry
model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
model = mlflow.sklearn.load_model(model_uri)

# Export as standalone pickle
joblib.dump(model, EXPORT_PATH)

print(f"Champion model loaded from: {model_uri}")
print(f"Model exported successfully to: {EXPORT_PATH}")