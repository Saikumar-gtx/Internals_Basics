import os
import json
import time
import threading
import joblib
import numpy as np
import pandas as pd
import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "tuned_model.pkl")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "step3_s4.json")
PORT = 8888

# ── Load model ────────────────────────────────────────────────────────────────
model = joblib.load(MODEL_PATH)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="PipeWatch API")


class PipeInput(BaseModel):
    pipe_age_years: float = Field(..., ge=1, le=40)
    flow_rate_lph: float = Field(..., ge=100, le=5000)
    moisture_pct: float = Field(..., ge=10, le=90)
    wall_thickness_mm: float = Field(..., ge=2, le=15)


@app.get("/status")
def status():
    return {"status": "operational", "service": "PipeWatch API"}


@app.post("/forecast")
def forecast(data: PipeInput):
    features = pd.DataFrame([{
        "pipe_age_years": data.pipe_age_years,
        "flow_rate_lph": data.flow_rate_lph,
        "moisture_pct": data.moisture_pct,
        "wall_thickness_mm": data.wall_thickness_mm,
    }])
    prediction = float(model.predict(features)[0])
    return {"prediction": round(prediction, 6)}


# ── Internal test + JSON generation ──────────────────────────────────────────
def _run_server():
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


def _test_and_save():
    base_url = f"http://localhost:{PORT}"

    # Wait until server is ready
    for _ in range(30):
        try:
            requests.get(f"{base_url}/status", timeout=1)
            break
        except Exception:
            time.sleep(0.5)

    health_response = requests.get(f"{base_url}/status").json()

    test_input = {
        "pipe_age_years": 20,
        "flow_rate_lph": 2004.1,
        "moisture_pct": 62.7,
        "wall_thickness_mm": 6.1,
    }
    forecast_response = requests.post(f"{base_url}/forecast", json=test_input).json()
    prediction = forecast_response["prediction"]

    print(f"Health: {health_response}")
    print(f"Test input: {test_input}")
    print(f"Prediction: {prediction}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    result = {
        "health_endpoint": "/status",
        "predict_endpoint": "/forecast",
        "port": PORT,
        "health_response": health_response,
        "test_input": test_input,
        "prediction": prediction,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()

    test_thread = threading.Thread(target=_test_and_save, daemon=True)
    test_thread.start()
    test_thread.join()

    print(f"\nPipeWatch API running on http://0.0.0.0:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")
