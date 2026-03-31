import mlflow
from mlflow.tracking import MlflowClient

# -----------------------------
# Config
# -----------------------------
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Credit Default Classification"
REGISTERED_MODEL_NAME = "CreditDefaultModel"

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# Get experiment
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")

experiment_id = experiment.experiment_id

# Search runs ordered by recall descending
runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.recall DESC"]
)

if runs.empty:
    raise ValueError("No runs found in the experiment.")

best_run = runs.iloc[0]
best_run_id = best_run["run_id"]
best_recall = best_run["metrics.recall"]
best_model_name = best_run["params.model_name"]

print("Best run selected:")
print(f"Run ID: {best_run_id}")
print(f"Model: {best_model_name}")
print(f"Recall: {best_recall}")

# Model URI from the best run
model_uri = f"runs:/{best_run_id}/model"

# Register model
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=REGISTERED_MODEL_NAME
)

print(f"Registered model name: {REGISTERED_MODEL_NAME}")
print(f"Registered version: {registered_model.version}")

# Assign Champion alias to this version
client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="Champion",
    version=registered_model.version sss
)

print(f"Alias 'Champion' assigned to version {registered_model.version}")