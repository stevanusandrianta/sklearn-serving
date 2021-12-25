from pydantic import BaseModel
from typing import Any, Optional
from fastapi import FastAPI

import numpy as np
import os
import joblib

MODEL_PATH = os.environ.get("MODEL_PATH")
MODEL = None

if MODEL_PATH:
    MODEL = joblib.load(MODEL_PATH)


class PredictionInput(BaseModel):
    data: list


class PredictionOutput(BaseModel):
    prediction: str
    probability: str


app = FastAPI()


@app.get("/")
def read_root():
    return {"status": "healthy"}


@app.post("/predict/")
def read_item(predictionInput: PredictionInput):
    if not MODEL:
        return {"error": "MODEL_PATH is not initialized"}

    prediction = MODEL.predict(np.array(predictionInput.data))
    probabilities = MODEL.predict_proba(np.array(predictionInput.data))

    return PredictionOutput(
        prediction=str(prediction),
        probability=str(probabilities)
    )
