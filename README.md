# Credit Default MLOps Pipeline

This project implements an end-to-end MLOps workflow for predicting credit card default risk.

## Features
- MLflow experiment tracking
- Automatic best model selection (based on recall)
- Model Registry with Champion alias
- Dockerized FastAPI inference API
- Kubernetes-ready deployment

## How to run

### Train models
python src/train_models.py

### Select best model
python src/select_best_model.py

### Export champion model
python src/export_champion.py

### Run API locally
uvicorn app:app --reload

### Run with Docker
docker build -t credit-default-inference .
docker run -p 8000:8000 credit-default-inference

## DVC Remote Setup (Portable)
This project expects a DVC remote URL from an environment variable instead of a machine-specific path.

Set your remote once before `dvc pull` or `dvc push`:

PowerShell:
`$env:DVC_REMOTE_URL="your-dvc-remote-url"`

Bash:
`export DVC_REMOTE_URL="your-dvc-remote-url"`
