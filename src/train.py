"""Train, compare, save and optionally register the best classifier with MLflow."""
from pathlib import Path
import json
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from feature_engineering import FEATURE_COLUMNS, TARGET
from preprocessing import build_preprocessor, create_feature_store

ROOT = Path(__file__).resolve().parents[1]

def train_and_evaluate():
    # Explicitly load the saved feature-store file to demonstrate feature reuse.
    feature_path = ROOT/"features/feature_store.csv"
    if not feature_path.exists():
        create_feature_store()
    data = pd.read_csv(feature_path)
    X, y = data[FEATURE_COLUMNS], data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=250, max_depth=8, min_samples_leaf=2,
            class_weight="balanced", random_state=42
        )
    }
    rows, fitted = [], {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", estimator)
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        }
        rows.append(row)
        fitted[name] = pipeline
    results = pd.DataFrame(rows).sort_values(
        ["recall", "f1_score", "accuracy"], ascending=False
    )
    best_name = results.iloc[0]["model"]
    best = fitted[best_name]
    (ROOT/"models").mkdir(exist_ok=True)
    (ROOT/"outputs").mkdir(exist_ok=True)
    joblib.dump(best, ROOT/"models/best_model.joblib")
    results.to_csv(ROOT/"outputs/model_comparison.csv", index=False)
    with open(ROOT/"outputs/model_metrics.json", "w") as f:
        json.dump({"selected_model": best_name, "metrics": rows}, f, indent=2)
    sample = X_test.head(5).copy()
    sample["actual"] = y_test.head(5).values
    sample["predicted"] = best.predict(X_test.head(5))
    sample.to_csv(ROOT/"outputs/sample_predictions.csv", index=False)

    # Local MLflow tracking and Model Registry. Requires MLflow installed.
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT/'mlflow.db').as_posix()}")
    mlflow.set_experiment("conveyor-belt-jam-prediction")
    with mlflow.start_run(run_name=best_name) as run:
        mlflow.log_param("selected_model", best_name)
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        best_row = results.iloc[0].to_dict()
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            mlflow.log_metric(metric, float(best_row[metric]))
        signature = infer_signature(X_train, best.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            best, name="conveyor_classifier", signature=signature,
            input_example=X_train.head(3),
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"]
        )
        try:
            registered = mlflow.register_model(
                model_uri=model_info.model_uri,
                name="ConveyorBeltJamClassifier"
            )
            print("Registered MLflow model version:", registered.version)
        except Exception as exc:
            print("MLflow model artifact logged, but registry registration needs attention:", exc)
        print("MLflow run ID:", run.info.run_id)

    print("\nModel comparison:\n", results.to_string(index=False))
    print("\nSelected model:", best_name)
    return best_name, results

if __name__ == "__main__":
    train_and_evaluate()
