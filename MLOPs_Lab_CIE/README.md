# Internals_Basics — MLOps CIE Lab

**Course:** MLOps (24AM6AEMLO) | **USN:** 1BM23AI166 | **Date:** 04 May 2026  
**Institution:** BMS College of Engineering, Department of Machine Learning

---

## Scenario

PipeWatch monitors pipeline infrastructure. This project trains and serves a model that predicts **corrosion risk scores** to help prioritize inspection and replacement schedules.

---

## Repository Structure

```
Internals_Basics/
└── MLOPs_Lab_CIE/
    ├── data/
    │   └── training_data.csv          # 25-row dataset (do not modify)
    ├── src/
    │   ├── train.py                   # Task 1 — Training + MLflow tracking
    │   ├── tune.py                    # Task 2 — Hyperparameter tuning
    │   ├── api.py                     # Task 3 — FastAPI serving
    │   └── register_model.py          # Task 4 — Model registration
    ├── models/
    │   ├── best_model.pkl             # Best model from Task 1 (generated)
    │   └── tuned_model.pkl            # Tuned model from Task 2 (generated)
    ├── results/
    │   ├── step1_s1.json              # Task 1 output
    │   ├── step2_s2.json              # Task 2 output
    │   ├── step3_s4.json              # Task 3 output
    │   └── step4_s6.json              # Task 4 output
    ├── requirements.txt
    └── .gitignore
```

---

## Dataset

**File:** `data/training_data.csv`

| Feature | Range | Description |
|---|---|---|
| `pipe_age_years` | 1 – 40 | Age of the pipe in years |
| `flow_rate_lph` | 100 – 5000 | Flow rate in litres per hour |
| `moisture_pct` | 10 – 90 | Moisture percentage |
| `wall_thickness_mm` | 2 – 15 | Wall thickness in millimetres |
| `corrosion_risk_score` | — | **Target variable** |

---

## Setup

### Prerequisites
- Python 3.9+
- pip

### Install dependencies

```bash
cd Internals_Basics/MLOPs_Lab_CIE
pip install -r requirements.txt
```

---

## Running the Project

> Run each task in order — each builds on the previous one.

### Task 1 — Experiment Tracking & Model Comparison

```bash
python src/train.py
```

**What it does:**
- Trains Ridge and GradientBoostingRegressor models
- Logs parameters, MAE, RMSE, and tag `priority=high` to MLflow
- Experiment name: `pipewatch-corrosion-risk-score`
- Selects best model by lowest RMSE
- Saves best model → `models/best_model.pkl`

**Output:** `results/step1_s1.json`
```json
{
  "experiment_name": "pipewatch-corrosion-risk-score",
  "models": [
    {"name": "Ridge", "mae": 7.245093, "rmse": 9.193681},
    {"name": "GradientBoosting", "mae": 8.711279, "rmse": 10.435461}
  ],
  "best_model": "Ridge",
  "best_metric_name": "rmse",
  "best_metric_value": 9.193681
}
```

---

### Task 2 — Hyperparameter Tuning

```bash
python src/tune.py
```

**What it does:**
- Runs GridSearchCV on GradientBoostingRegressor with 3-fold CV
- Parameter grid: `n_estimators` [50, 150] × `learning_rate` [0.05, 0.1, 0.2] × `max_depth` [3, 5, 10] → 18 total trials
- Logs each trial as a nested MLflow run under parent `tuning-pipewatch`
- Saves best tuned model → `models/tuned_model.pkl`

**Output:** `results/step2_s2.json`
```json
{
  "search_type": "grid",
  "n_folds": 3,
  "total_trials": 18,
  "best_params": {"learning_rate": 0.05, "max_depth": 3, "n_estimators": 50},
  "best_mae": 9.081767,
  "best_cv_mae": 16.035407,
  "parent_run_name": "tuning-pipewatch"
}
```

---

### Task 3 — FastAPI Serving

```bash
python src/api.py
```

**What it does:**
- Loads `models/tuned_model.pkl`
- Starts FastAPI server on port `8888`
- Automatically calls both endpoints internally, saves the results JSON, then keeps the server running
- Press `Ctrl+C` to stop

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/status` | Health check |
| `POST` | `/forecast` | Predict corrosion risk score |

**GET /status — example response:**
```json
{"status": "operational", "service": "PipeWatch API"}
```

**POST /forecast — example request:**
```json
{
  "pipe_age_years": 20,
  "flow_rate_lph": 2004.1,
  "moisture_pct": 62.7,
  "wall_thickness_mm": 6.1
}
```

**Input validation (Pydantic):** All fields are required and range-validated. Invalid inputs return HTTP `422`.

**Output:** `results/step3_s4.json`
```json
{
  "health_endpoint": "/status",
  "predict_endpoint": "/forecast",
  "port": 8888,
  "health_response": {"status": "operational", "service": "PipeWatch API"},
  "test_input": {"pipe_age_years": 20, "flow_rate_lph": 2004.1, "moisture_pct": 62.7, "wall_thickness_mm": 6.1},
  "prediction": 38.553773
}
```

---

### Task 4 — Model Registration

```bash
python src/register_model.py
```

**What it does:**
- Registers the tuned model in the MLflow Model Registry
- Registered name: `pipewatch-corrosion-risk-score-predictor`
- Retrieves version number and run ID
- Links back to the RMSE from Task 1

**Output:** `results/step4_s6.json`
```json
{
  "registered_model_name": "pipewatch-corrosion-risk-score-predictor",
  "version": 1,
  "run_id": "ee5817628c6447b0a5755b4a37de3e87",
  "source_metric": "rmse",
  "source_metric_value": 9.193681
}
```

---

## MLflow UI

To inspect all runs and the model registry visually:

```bash
cd Internals_Basics/MLOPs_Lab_CIE
mlflow ui
```

Open `http://localhost:5000` in your browser.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `scikit-learn` | Model training, GridSearchCV |
| `mlflow` | Experiment tracking, model registry |
| `fastapi` | REST API server |
| `uvicorn` | ASGI server for FastAPI |
| `pydantic` | Input validation |
| `joblib` | Model serialisation |
| `pandas` / `numpy` | Data handling |
| `requests` | Internal API test calls |
