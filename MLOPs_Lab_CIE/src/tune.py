import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

    # Load dataset
    data_path = os.path.join(BASE_DIR, "data", "training_data.csv")
    df = pd.read_csv(data_path)
    X = df.drop("corrosion_risk_score", axis=1)
    y = df["corrosion_risk_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # The provided param grid (n_estimators, learning_rate, max_depth) is
    # specific to GradientBoostingRegressor, so we tune it unconditionally.
    step1_path = os.path.join(BASE_DIR, "results", "step1_s1.json")
    with open(step1_path) as f:
        step1 = json.load(f)
    print(f"Best model from Task 1: {step1['best_model']} — tuning GradientBoosting (param grid is GB-specific)")

    param_grid = {
        "n_estimators": [50, 150],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, 10],
    }
    total_trials = (
        len(param_grid["n_estimators"])
        * len(param_grid["learning_rate"])
        * len(param_grid["max_depth"])
    )

    experiment_name = "pipewatch-corrosion-risk-score"
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)

    gs = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )

    with mlflow.start_run(run_name="tuning-pipewatch") as parent_run:
        gs.fit(X_train, y_train)

        # Log each trial as a nested run
        for i, params in enumerate(gs.cv_results_["params"]):
            with mlflow.start_run(run_name=f"trial-{i + 1}", nested=True):
                mlflow.log_params(params)
                cv_rmse = float(-gs.cv_results_["mean_test_score"][i])
                mlflow.log_metric("cv_rmse", round(cv_rmse, 6))

        # Log best config in parent run
        mlflow.log_params(gs.best_params_)
        best_cv_rmse = float(-gs.best_score_)
        mlflow.log_metric("best_cv_rmse", round(best_cv_rmse, 6))

        # Compute CV MAE for best estimator
        cv_mae_scores = cross_val_score(
            gs.best_estimator_, X_train, y_train,
            cv=3, scoring="neg_mean_absolute_error"
        )
        best_cv_mae = float(-cv_mae_scores.mean())

        # Test-set MAE
        test_preds = gs.predict(X_test)
        best_mae = float(mean_absolute_error(y_test, test_preds))
        best_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))

        mlflow.log_metric("test_mae", round(best_mae, 6))
        mlflow.log_metric("test_rmse", round(best_rmse, 6))
        mlflow.sklearn.log_model(gs.best_estimator_, "tuned_model")

        parent_run_id = parent_run.info.run_id

    print(f"Best params: {gs.best_params_}")
    print(f"Best CV RMSE: {best_cv_rmse:.4f}")
    print(f"Test MAE: {best_mae:.4f}  Test RMSE: {best_rmse:.4f}")

    # Save tuned model
    tuned_path = os.path.join(BASE_DIR, "models", "tuned_model.pkl")
    joblib.dump(gs.best_estimator_, tuned_path)
    print(f"Saved tuned model to {tuned_path}")

    # Save run_id for register_model.py
    run_id_path = os.path.join(BASE_DIR, "results", ".tune_run_id")
    with open(run_id_path, "w") as f:
        f.write(parent_run_id)

    step2 = {
        "search_type": "grid",
        "n_folds": 3,
        "total_trials": total_trials,
        "best_params": gs.best_params_,
        "best_mae": round(best_mae, 6),
        "best_cv_mae": round(best_cv_mae, 6),
        "parent_run_name": "tuning-pipewatch",
    }

    out_path = os.path.join(BASE_DIR, "results", "step2_s2.json")
    with open(out_path, "w") as f:
        json.dump(step2, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
