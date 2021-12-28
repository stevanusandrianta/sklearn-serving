from pydantic import BaseModel
from typing import Any, Optional
from fastapi import FastAPI

import numpy as np
import pandas as pd
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

    if type(predictionInput.data[0]) == dict:
        prediction_data = pd.DataFrame(predictionInput.data)
    else:
        prediction_data = np.array(predictionInput.data)

    prediction = MODEL.predict(prediction_data)
    probabilities = MODEL.predict_proba(prediction_data)

    return PredictionOutput(
        prediction=str(prediction),
        probability=str(probabilities)
    )
