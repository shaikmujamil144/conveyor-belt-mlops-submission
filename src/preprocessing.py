"""Build and run the reusable preprocessing + feature-store pipeline."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_engineering import FEATURE_COLUMNS, engineer_features

ROOT = Path(__file__).resolve().parents[1]

def build_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return ColumnTransformer([("numeric", numeric, FEATURE_COLUMNS)])

def create_feature_store():
    v1 = pd.read_csv(ROOT/"datasets/conveyor_v1.csv")
    v2 = pd.read_csv(ROOT/"datasets/conveyor_v2.csv")
    v1["dataset_version"] = "v1_normal"
    v2["dataset_version"] = "v2_festival"
    combined = pd.concat([v1, v2], ignore_index=True)
    engineered = engineer_features(combined)
    (ROOT/"features").mkdir(exist_ok=True)
    engineered.to_csv(ROOT/"features/feature_store.csv", index=False)
    print("Feature store created:", ROOT/"features/feature_store.csv")
    print("Rows:", len(engineered))
    return engineered

if __name__ == "__main__":
    create_feature_store()
