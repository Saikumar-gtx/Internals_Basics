import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

    data_path = os.path.join(BASE_DIR, "data", "training_data.csv")
    df = pd.read_csv(data_path)
    X = df.drop("corrosion_risk_score", axis=1)
    y = df["corrosion_risk_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    experiment_name = "pipewatch-corrosion-risk-score"
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)

    candidates = {
        "Ridge": {
            "model": Ridge(alpha=1.0),
            "params": {"alpha": 1.0},
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
            ),
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42,
            },
        },
    }

    results = []

    for name, cfg in candidates.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("priority", "high")
            mlflow.log_params(cfg["params"])

            model = cfg["model"]
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mae = float(mean_absolute_error(y_test, preds))
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(model, "model")

            results.append({"name": name, "mae": round(mae, 6), "rmse": round(rmse, 6), "model_obj": model})
            print(f"{name} — MAE: {mae:.4f}  RMSE: {rmse:.4f}")

    best = min(results, key=lambda x: x["rmse"])
    best_name = best["name"]
    best_rmse = best["rmse"]
    best_mae = best["mae"]

    model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
    joblib.dump(best["model_obj"], model_path)
    print(f"\nBest model: {best_name} (RMSE={best_rmse})")
    print(f"Saved to {model_path}")

    step1 = {
        "experiment_name": experiment_name,
        "models": [{"name": r["name"], "mae": r["mae"], "rmse": r["rmse"]} for r in results],
        "best_model": best_name,
        "best_metric_name": "rmse",
        "best_metric_value": best_rmse,
    }

    out_path = os.path.join(BASE_DIR, "results", "step1_s1.json")
    with open(out_path, "w") as f:
        json.dump(step1, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
