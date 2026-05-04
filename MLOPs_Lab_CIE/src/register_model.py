import os
import json
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REGISTERED_MODEL_NAME = "pipewatch-corrosion-risk-score-predictor"
EXPERIMENT_NAME = "pipewatch-corrosion-risk-score"


def main():
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

    mlflow.set_tracking_uri("file:./mlruns")

    client = MlflowClient()

    # ── Locate the tuning parent run ─────────────────────────────────────────
    run_id_file = os.path.join(BASE_DIR, "results", ".tune_run_id")
    if os.path.exists(run_id_file):
        with open(run_id_file) as f:
            parent_run_id = f.read().strip()
        print(f"Using tune run_id from file: {parent_run_id}")
    else:
        # Fallback: search by run name
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.`mlflow.runName` = 'tuning-pipewatch'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError("Could not find 'tuning-pipewatch' run. Run tune.py first.")
        parent_run_id = runs[0].info.run_id
        print(f"Found tune run_id via search: {parent_run_id}")

    model_uri = f"runs:/{parent_run_id}/tuned_model"
    print(f"Registering model from: {model_uri}")

    # ── Register model ────────────────────────────────────────────────────────
    mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    version = int(mv.version)
    run_id = mv.run_id
    print(f"Registered '{REGISTERED_MODEL_NAME}' — version={version}, run_id={run_id}")

    # ── Retrieve best RMSE from Task 1 ────────────────────────────────────────
    step1_path = os.path.join(BASE_DIR, "results", "step1_s1.json")
    with open(step1_path) as f:
        step1 = json.load(f)
    source_metric_value = step1["best_metric_value"]

    # ── Save JSON ─────────────────────────────────────────────────────────────
    step4 = {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "version": version,
        "run_id": run_id,
        "source_metric": "rmse",
        "source_metric_value": source_metric_value,
    }

    out_path = os.path.join(BASE_DIR, "results", "step4_s6.json")
    with open(out_path, "w") as f:
        json.dump(step4, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
