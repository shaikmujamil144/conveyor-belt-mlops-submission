import sys
from pathlib import Path
import pandas as pd
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"api"))
from feature_engineering import engineer_features, FEATURE_COLUMNS
from app import ConveyorInput, predict, model

def valid_payload():
    return {
        "motor_current": 12.0, "rated_motor_current": 18.0,
        "vibration_level": 4.2, "belt_age": 2.5,
        "parcels_per_minute": 110.0, "dust_accumulation_index": 35.0,
        "days_since_maintenance": 20, "humidity": 55.0
    }

def test_feature_engineering_creates_required_columns():
    frame = engineer_features(pd.DataFrame([valid_payload()]))
    assert "motor_load_ratio" in frame.columns
    assert "belt_wear_pressure_index" in frame.columns
    assert frame.loc[0, "motor_load_ratio"] == pytest.approx(12/18)

def test_negative_sensor_input_is_rejected():
    bad = valid_payload()
    bad["motor_current"] = -1
    with pytest.raises(Exception):
        ConveyorInput(**bad)

def test_prediction_returns_valid_label():
    if model is None:
        pytest.skip("Train model before running API prediction test")
    result = predict(ConveyorInput(**valid_payload()))
    assert result["prediction"] in ["Jam Risk", "Normal Operation"]
    assert 0 <= result["jam_probability"] <= 1
