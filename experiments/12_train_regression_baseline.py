# experiments/12_train_regression_baseline.py
"""Experiment 12: Train regression baseline and save artifacts."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODELS_DIR = Path("results/models")
MODEL_PATH = MODELS_DIR / "regression_model.pkl"
SCALER_PATH = MODELS_DIR / "regression_scaler.pkl"
WEIGHTS_PATH = MODELS_DIR / "regression_weights.json"
METADATA_PATH = MODELS_DIR / "regression_metadata.json"
TABLES_DIR = Path("results/tables")
QUALITY_CSV_PATH = TABLES_DIR / "regression_quality_metrics.csv"
CSV_HEADERS = [
    "mode",
    "mae",
    "mse",
    "rmse",
    "r2",
    "mean_abs_diff_vs_plaintext",
    "max_abs_diff_vs_plaintext",
    "mean_abs_diff_vs_encoded",
    "max_abs_diff_vs_encoded",
    "match_rate_vs_plaintext_tol_1e_2",
    "match_rate_vs_encoded_tol_1e_2",
]


def _compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute scalar regression metrics."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": mse**0.5,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _write_baseline_quality_row(metrics: dict[str, float]) -> None:
    """Write regression quality metrics CSV with header and one baseline row."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    row: dict[str, float | str] = {
        "mode": "Plaintext baseline",
        "mae": metrics["mae"],
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "mean_abs_diff_vs_plaintext": 0.0,
        "max_abs_diff_vs_plaintext": 0.0,
        "mean_abs_diff_vs_encoded": 0.0,
        "max_abs_diff_vs_encoded": 0.0,
        "match_rate_vs_plaintext_tol_1e_2": 1.0,
        "match_rate_vs_encoded_tol_1e_2": 1.0,
    }
    with QUALITY_CSV_PATH.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """Train Ridge regression baseline, evaluate, and save artifacts."""
    logger.info("Starting regression baseline training...")

    dataset = load_diabetes(as_frame=True)
    features = dataset.data
    target = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    metrics = _compute_metrics(y_true=y_test, y_pred=y_pred)
    logger.info("Regression test metrics: %s", json.dumps(metrics, ensure_ascii=False))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(pipeline.named_steps["scaler"], SCALER_PATH)

    regressor = pipeline.named_steps["regressor"]
    weights_payload = {
        "w": regressor.coef_.tolist(),
        "b": float(regressor.intercept_),
    }
    WEIGHTS_PATH.write_text(
        json.dumps(weights_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metadata_payload = {
        "dataset": "load_diabetes",
        "feature_names": features.columns.tolist(),
        "target_name": dataset.target.name if hasattr(dataset.target, "name") else "target",
        "test_size": 0.2,
        "random_state": 42,
        "model": "Ridge",
        "alpha": 1.0,
    }
    METADATA_PATH.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _write_baseline_quality_row(metrics)

    logger.info("Saved model artifacts to %s", MODELS_DIR)
    logger.info("Saved regression quality metrics to %s", QUALITY_CSV_PATH)
    logger.info("Regression baseline training completed.")


if __name__ == "__main__":
    main()
