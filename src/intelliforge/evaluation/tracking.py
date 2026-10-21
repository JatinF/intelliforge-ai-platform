"""Local run record.

The shape matches the params and metrics a tracking client would store.
The MLflow client is not imported unless INTELLIFORGE_MLFLOW=1, and that path is not used by the tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def write_run(path: Path, params: dict, metrics: dict) -> dict:
    record = {
        "tracking": "local-json",
        "mlflow_client": "not-run",
        "params": params,
        "metrics": metrics,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    return record


def log_mlflow(params: dict, metrics: dict, tracking_uri: str) -> None:
    if os.environ.get("INTELLIFORGE_MLFLOW") != "1":
        raise RuntimeError("INTELLIFORGE_MLFLOW is not 1")
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run():
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
