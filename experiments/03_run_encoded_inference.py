# experiments/03_run_encoded_inference.py
"""Experiment 03: Evaluate encoded plaintext inference quality."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from app.config import RANDOM_STATE, SCALE, TEST_SIZE
from app.data import load_dataset, split_dataset
from app.inference import encoded_plaintext_inference, plaintext_inference
from app.metrics import classification_metrics
from app.model import extract_linear_params, load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


MODEL_PATH = Path("results/models/model.pkl")
TABLES_DIR = Path("results/tables")
QUALITY_CSV_PATH = TABLES_DIR / "quality_metrics.csv"
CSV_HEADERS = [
    "mode",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "match_rate",
]


def _append_quality_row(row: dict[str, float | str]) -> None:
    """
    Append one quality row to CSV, creating file with headers if needed.

    Args:
        row: Row values keyed by CSV header names.
    """

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = QUALITY_CSV_PATH.exists()

    with QUALITY_CSV_PATH.open("a", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """
    Run encoded plaintext inference and persist quality metrics.
    """

    logger.info("Running encoded plaintext inference...")

    model = load_model(str(MODEL_PATH))
    features, target = load_dataset()
    _, x_test, _, y_test = split_dataset(
        features=features,
        target=target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    baseline_pred, _ = plaintext_inference(model=model, x_test=x_test)

    scaler = model.named_steps["scaler"]
    x_scaled = scaler.transform(x_test)
    w, b = extract_linear_params(model)

    encoded_pred, encoded_prob = encoded_plaintext_inference(
        x_scaled=x_scaled,
        w=w,
        b=b,
        scale=SCALE,
        threshold=0.5,
    )

    metrics = classification_metrics(
        y_true=y_test.to_numpy(), y_pred=encoded_pred, y_prob=encoded_prob
    )
    match_rate = float((baseline_pred == encoded_pred).mean())

    logger.info("Encoded metrics: %s", metrics)
    logger.info("Match rate vs baseline: %.6f", match_rate)

    row = {
        "mode": "Encoded plaintext",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics.get("roc_auc", float("nan")),
        "match_rate": match_rate,
    }
    _append_quality_row(row)

    logger.info("Saved encoded quality row to %s", QUALITY_CSV_PATH)
    logger.info("Encoded plaintext inference completed.")


if __name__ == "__main__":
    main()
