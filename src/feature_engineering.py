"""Feature definitions shared by training and inference."""
import pandas as pd

RAW_FEATURES = [
    "motor_current", "rated_motor_current", "vibration_level", "belt_age",
    "parcels_per_minute", "dust_accumulation_index",
    "days_since_maintenance", "humidity"
]
FEATURE_COLUMNS = RAW_FEATURES + ["motor_load_ratio", "belt_wear_pressure_index"]
TARGET = "next_shift_jam_status"

def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["motor_load_ratio"] = (
        data["motor_current"] / data["rated_motor_current"].replace(0, float("nan"))
    )
    data["belt_wear_pressure_index"] = (
        (data["belt_age"] / 5.0) * 0.30
        + (data["vibration_level"] / 10.0) * 0.30
        + (data["dust_accumulation_index"] / 100.0) * 0.20
        + (data["days_since_maintenance"] / 90.0) * 0.20
    )
    return data
