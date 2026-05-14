# experiments/01_train_baseline.py
"""Experiment 01: Train baseline model and save artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from app.config import RANDOM_STATE, TEST_SIZE
from app.data import load_dataset, split_dataset
from app.model import extract_linear_params, save_model, train_baseline_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODELS_DIR = Path("results/models")
MODEL_PATH = MODELS_DIR / "model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
WEIGHTS_PATH = MODELS_DIR / "weights.json"
METADATA_PATH = MODELS_DIR / "metadata.json"


def _compute_metrics(y_true: Any, y_pred: Any, y_proba: Any) -> dict[str, float]:
    """Compute baseline classification metrics.

    Args:
        y_true: Ground-truth labels for the test set.
        y_pred: Predicted class labels for the test set.
        y_proba: Predicted positive-class probabilities for the test set.

    Returns:
        Dictionary with scalar metric values.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def main() -> None:
    """
    Run baseline training, evaluation, and artifact persistence.
    """
    logger.info("Starting baseline training...")

    features, target = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(
        features=features,
        target=target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    pipeline = train_baseline_model(
        x_train=x_train,
        y_train=y_train,
        random_state=RANDOM_STATE,
    )

    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)[:, 1]
    metrics = _compute_metrics(y_true=y_test, y_pred=y_pred, y_proba=y_proba)
    logger.info("Test metrics: %s", json.dumps(metrics, ensure_ascii=False))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    save_model(pipeline=pipeline, filepath=str(MODEL_PATH))
    joblib.dump(pipeline.named_steps["scaler"], SCALER_PATH)

    w, b = extract_linear_params(pipeline)
    weights_payload = {"w": w.tolist(), "b": float(b)}
    WEIGHTS_PATH.write_text(
        json.dumps(weights_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dataset = load_breast_cancer()
    metadata_payload = {
        "feature_names": features.columns.tolist(),
        "classes": dataset.target_names.tolist(),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Saved model artifacts to %s", MODELS_DIR)
    logger.info("Baseline training completed.")


if __name__ == "__main__":
    main()
