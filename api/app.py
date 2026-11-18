"""FastAPI inference service for conveyor-belt jam risk."""
from pathlib import Path
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from feature_engineering import FEATURE_COLUMNS, engineer_features

MODEL_PATH = ROOT/"models/best_model.joblib"
app = FastAPI(title="Conveyor Belt Jam Prediction API", version="1.0.0")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

class ConveyorInput(BaseModel):
    motor_current: float = Field(gt=0, le=100)
    rated_motor_current: float = Field(gt=0, le=100)
    vibration_level: float = Field(ge=0, le=50)
    belt_age: float = Field(ge=0, le=30)
    parcels_per_minute: float = Field(ge=0, le=500)
    dust_accumulation_index: float = Field(ge=0, le=100)
    days_since_maintenance: int = Field(ge=0, le=3650)
    humidity: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def motor_current_not_implausible(self):
        if self.motor_current > self.rated_motor_current * 2:
            raise ValueError("motor_current cannot exceed twice rated_motor_current")
        return self

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(payload: ConveyorInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact not found. Run src/train.py first.")
    raw = pd.DataFrame([payload.model_dump()])
    features = engineer_features(raw)[FEATURE_COLUMNS]
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    return {
        "prediction": "Jam Risk" if prediction == 1 else "Normal Operation",
        "jam_probability": round(probability, 4),
        "predicted_class": prediction
    }
