import joblib
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from src.api.schemas import CustomerInput

# Get the directory where app.py is located
BASE_DIR = Path(__file__).resolve().parents[1]

# Build the path to the model: go up one level to 'src', then into 'artifacts'
model_path = BASE_DIR.parent / "artifacts" / "model.joblib"
threshold_path = BASE_DIR.parent / "artifacts" / "threshold.txt"

app = FastAPI(title="Customer Churn Prediction API")

# Load Artifacts

model = joblib.load(model_path)
threshold  = float(open(threshold_path).read())

@app.get("/")

def health_check():
    return {"status": "API Running"}

@app.post("/predict")
def predict_churn(customer: CustomerInput):
    data = pd.DataFrame([customer.model_dump()])
    prob = model.predict_proba(data)[0][1]
    churn = int(prob >= threshold)
    
    return {
        "churn_prediction": round(prob, 4),
        "churn_probability": churn
    }