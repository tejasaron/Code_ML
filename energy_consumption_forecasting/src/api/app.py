from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


from src.models.model_io import load_model

app = FastAPI ( title = "Energy Consumption Forecasting API",
    description = "Predicts short term energy demand for smart grids",
    version='1.0')

# Load model at startup and log details

try:
    model = load_model("aep_random_forest")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


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

    try:
        logger.info(f"Received prediction request: {features.dict()}")

        data = pd.DataFrame([features.__dict__])
        prediction = model.predict(data)[0]

        logger.info(f"Prediction successful: {prediction:.2f} MW")

        return {
            "predicted_load_mw": round(float(prediction), 2)
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {
            "error": "Prediction failed. Check logs for details."
        }

