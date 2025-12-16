from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd


from src.models import model_io

app = FastAPI ( title = "Energy Consumption Forecasting API",
    description = "Predicts short term energy demand for smart grids",
    version='1.0')

# Load model at startup

model = model_io.load_model('aep_random_forest')


class EnergyFeatures(BaseModel):
    hour: int
    day_of_week: int
    month: int
    lag_1: float
    lag_24: float
    lag_168: float
    rolling_mean_24: float
    rolling_mean_168: float

@app.post("/predict")

def predict_energy_load(features: EnergyFeatures):
    data = pd.DataFrame([features.__dict__])
    prediction = model.predict(data)[0]

    return { "predicted_load_mw": round(float(prediction), 2)}

