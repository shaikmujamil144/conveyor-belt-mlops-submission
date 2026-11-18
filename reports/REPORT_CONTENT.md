# MLOps Implementation Report
## Predicting Conveyor Belt Jamming in an Automated Parcel Sorting Centre

### 1. Abstract
This project implements an end-to-end academic MLOps workflow for predicting next-shift conveyor-belt jamming. Synthetic sensor and maintenance records are generated in two versions: normal operations and a festival shopping period. A preprocessing pipeline imputes missing numerical values and scales features. Two classifiers are trained and compared. The selected pipeline is saved as an artifact, tracked and registered using MLflow, tested through pytest and GitHub Actions, and exposed through a FastAPI service.

### 2. Problem Statement
Motor overload, abnormal vibration, dust accumulation, belt age, high parcel throughput and delayed maintenance can contribute to conveyor interruptions. The system predicts `next_shift_jam_status` from the current operating conditions. Class 1 means Jam; class 0 means No Jam.

### 3. Objectives
- Generate at least 250 synthetic shift records per dataset version.
- Maintain normal and festival-period dataset versions.
- Build an automated preprocessing and feature-engineering pipeline.
- Store and reuse engineered features.
- Train Logistic Regression and Random Forest classifiers.
- Compare accuracy, recall, F1-score and confusion matrices.
- Register the selected model using MLflow.
- Automate tests using GitHub Actions.
- Serve predictions using FastAPI `/predict` and `/health` endpoints.

### 4. System Architecture
Data generation → dataset versioning → feature engineering → feature store → preprocessing pipeline → model training/evaluation → MLflow tracking and registry → automated tests (GitHub Actions) → FastAPI model serving.

### 5. Dataset Creation and Versioning
Each version contains 350 records and the same nine columns. Version 1 simulates ordinary operating conditions. Version 2 simulates a festival-shopping period with a 30% higher mean parcel throughput and a higher simulated risk term. A small amount of sensor missingness is inserted to demonstrate imputation. The target is generated probabilistically from simulated operating-stress variables.

### 6. Data Pipeline and Feature Store
The feature engineering module creates:
- `motor_load_ratio`: motor current divided by rated motor current.
- `belt_wear_pressure_index`: a weighted composite of normalized belt age, vibration, dust and days since maintenance.

The engineered rows are saved to `features/feature_store.csv`. The training script reads this stored file rather than recomputing features from raw CSVs, demonstrating feature reuse. A scikit-learn pipeline applies median imputation and standard scaling.

### 7. Model Training and Evaluation
A stratified train/test split is used. Logistic Regression and Random Forest are trained using the same preprocessing pipeline. Accuracy, precision, recall, F1-score and confusion-matrix counts (TN, FP, FN, TP) are recorded in `outputs/model_comparison.csv`. The selection rule prioritizes recall, then F1-score, then accuracy because failing to identify a jam is an important error in this scenario.

### 8. MLflow Registry
The training script configures a local SQLite MLflow backend, logs parameters and metrics, logs the fitted scikit-learn pipeline with a model signature and attempts registration under `ConveyorBeltJamClassifier`. The actual run ID and registered model version are printed by the training script. Capture the MLflow UI after running locally and insert the screenshot in the final evidence section.

### 9. CI/CD
The workflow in `.github/workflows/ci.yml` installs dependencies, generates data, creates the feature store, trains the model and runs pytest. It includes a negative motor-current validation test and a model-prediction test. The GitHub Actions result is only available after pushing the repository to GitHub; capture the real successful Actions run screenshot after execution.

### 10. Model Serving
FastAPI exposes:
- `GET /health`: returns API health and whether the model artifact loaded.
- `POST /predict`: accepts sensor readings and returns `Jam Risk` or `Normal Operation`, predicted class and estimated jam probability.

The API validates sensor ranges and rejects negative motor current. Swagger documentation is available at `/docs`.

### 11. Limitations
All data is synthetic and the label is created from an illustrative simulation formula. Metrics are therefore demonstrations of the workflow, not proof of real-world predictive accuracy. Real deployment requires real sensor history, validation across time and machines, monitoring for drift, and domain-expert review.

### 12. Conclusion
The project demonstrates a reproducible MLOps lifecycle from synthetic dataset generation and feature reuse to model comparison, MLflow registry, automated testing and API serving.
