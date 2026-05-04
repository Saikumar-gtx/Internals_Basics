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
    │   └── training_data.csv          # 25-row dataset (do not modify) ✅ tracked
    ├── src/
    │   ├── train.py                   # Task 1 — Training + MLflow tracking
    │   ├── tune.py                    # Task 2 — Hyperparameter tuning
    │   ├── api.py                     # Task 3 — FastAPI serving
    │   └── register_model.py          # Task 4 — Model registration
    ├── models/
    │   ├── best_model.pkl             # ❌ git-ignored (generated locally)
    │   └── tuned_model.pkl            # ❌ git-ignored (generated locally)
    ├── results/
    │   ├── step1_s1.json              # ✅ tracked — Task 1 proof of execution
    │   ├── step2_s2.json              # ✅ tracked — Task 2 proof of execution
    │   ├── step3_s4.json              # ✅ tracked — Task 3 proof of execution
    │   └── step4_s6.json              # ✅ tracked — Task 4 proof of execution
    ├── mlruns/                        # ❌ git-ignored (MLflow local store)
    ├── .gitignore
    ├── README.md
    └── requirements.txt
```

### What is tracked vs ignored

| Path | Tracked | Reason |
|---|---|---|
| `data/training_data.csv` | ✅ Yes | Required input — must not be modified |
| `src/*.py` | ✅ Yes | All source code |
| `results/*.json` | ✅ Yes | Proof of execution — zero marks without these |
| `requirements.txt` | ✅ Yes | Dependency spec |
| `README.md` | ✅ Yes | Documentation |
| `models/*.pkl` | ❌ No | Binary files — regenerate by running scripts |
| `mlruns/` | ❌ No | Local MLflow store — not portable |
| `.venv/`, `venv/`, `env/` | ❌ No | Virtual environments |
| `__pycache__/`, `*.pyc` | ❌ No | Python bytecode |
| `.vscode/`, `.idea/` | ❌ No | IDE config |
| `*.log`, `*.tmp` | ❌ No | Temporary files |
| `.DS_Store`, `Thumbs.db` | ❌ No | OS-generated junk |

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

> Do not modify this file. It is version-controlled and must remain unchanged.

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
- Trains Ridge and GradientBoostingRegressor with `test_size=0.2`, `random_state=42`
- Logs parameters, MAE, RMSE, and tag `priority=high` to MLflow
- MLflow experiment name: `pipewatch-corrosion-risk-score`
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
- Automatically hits both endpoints, saves the results JSON, then keeps the server running
- Press `Ctrl+C` to stop the server

**Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | Health check |
| `POST` | `/forecast` | Predict corrosion risk score |

**GET /status**
```json
{"status": "operational", "service": "PipeWatch API"}
```

**POST /forecast — request body:**
```json
{
  "pipe_age_years": 20,
  "flow_rate_lph": 2004.1,
  "moisture_pct": 62.7,
  "wall_thickness_mm": 6.1
}
```

Input validation rules (Pydantic):

| Field | Min | Max |
|---|---|---|
| `pipe_age_years` | 1 | 40 |
| `flow_rate_lph` | 100 | 5000 |
| `moisture_pct` | 10 | 90 |
| `wall_thickness_mm` | 2 | 15 |

Out-of-range inputs return HTTP `422 Unprocessable Entity`.

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
- Registered model name: `pipewatch-corrosion-risk-score-predictor`
- Retrieves the assigned version number and run ID
- Links back to the best RMSE from Task 1

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

## Regenerating Models After Cloning

Since `models/*.pkl` is git-ignored, run the scripts in order after cloning:

```bash
pip install -r requirements.txt
python src/train.py
python src/tune.py
python src/api.py        # Ctrl+C after JSON is saved
python src/register_model.py
```

The `results/*.json` files are already committed and serve as proof of execution.

---

## MLflow UI

To browse all runs and the model registry:

```bash
cd Internals_Basics/MLOPs_Lab_CIE
mlflow ui
```

Open `http://localhost:5000` in your browser.

> `mlruns/` is git-ignored, so the UI is only available locally after running the scripts.

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
