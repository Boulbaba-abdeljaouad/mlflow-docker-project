# Credit Default MLOps Pipeline

This project implements an end-to-end MLOps workflow for predicting credit card default risk.

## Features
- MLflow experiment tracking
- Automatic best model selection (based on recall)
- Model Registry with Champion alias
- Dockerized FastAPI inference API
- Kubernetes-ready deployment

## How to run

### Prerequisites
- Python 3.11
- Docker
- DVC with Google Drive support: `pip install "dvc[gdrive]"`

### Pull dataset (required before training)
`dvc pull data/UCI_Credit_Card.csv.dvc`

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

## CI testing (GitHub Actions)
The workflows in this repository require these GitHub repository secrets:
- `DVC_REMOTE_URL`
- `GDRIVE_CREDENTIALS_DATA`
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
