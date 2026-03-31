import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

model = joblib.load("model.pkl")
app = FastAPI(title="Credit Default Champion Inference API")

class PredictionRequest(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "Credit Default inference API is running."}

@app.post("/predict")
def predict(request: PredictionRequest):
    if len(request.features) != 23:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 23 features, got {len(request.features)}"
        )

    features = np.array(request.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return {"prediction": int(prediction)}sdsds