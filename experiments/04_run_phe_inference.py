# experiments/04_run_phe_inference.py
"""Experiment 04: Run PHE inference on test set and measure quality."""

from __future__ import annotations

import csv
import logging
import warnings
from pathlib import Path

from app.client import Client
from app.config import KEY_LENGTH, RANDOM_STATE, SCALE, TEST_SIZE
from app.data import load_dataset, split_dataset
from app.encoding import encode_bias, encode_weights
from app.inference import encoded_plaintext_inference, phe_inference_batch, plaintext_inference
from app.metrics import classification_metrics
from app.model import extract_linear_params, load_model
from app.server import Server

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


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
    """Run PHE inference and persist quality metrics."""
    logger.info("Starting PHE inference...")

    model = load_model(str(MODEL_PATH))
    features, target = load_dataset()
    _, x_test, _, y_test = split_dataset(
        features=features,
        target=target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    scaler = model.named_steps["scaler"]
    w, b = extract_linear_params(model)

    client = Client(scaler=scaler, scale=SCALE, key_length=KEY_LENGTH)
    server = Server(
        w_int=encode_weights(w=w, scale=SCALE),
        b_int=encode_bias(b=b, scale=SCALE),
        public_key=client.public_key,
    )

    # Full test split is used; if runtime becomes a bottleneck, switch to a deterministic subset.
    phe_pred, phe_prob = phe_inference_batch(client=client, server=server, x_raw=x_test.to_numpy())

    _, _ = plaintext_inference(model=model, x_test=x_test)
    x_scaled = scaler.transform(x_test)
    encoded_pred, _ = encoded_plaintext_inference(
        x_scaled=x_scaled,
        w=w,
        b=b,
        scale=SCALE,
        threshold=0.5,
    )

    metrics = classification_metrics(y_true=y_test.to_numpy(), y_pred=phe_pred, y_prob=phe_prob)
    match_rate = float((encoded_pred == phe_pred).mean())

    logger.info("PHE metrics: %s", metrics)
    logger.info("Match rate vs encoded plaintext: %.6f", match_rate)

    row = {
        "mode": "PHE inference",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics.get("roc_auc", float("nan")),
        "match_rate": match_rate,
    }
    _append_quality_row(row)

    logger.info("Saved PHE quality row to %s", QUALITY_CSV_PATH)
    logger.info("PHE inference completed.")


if __name__ == "__main__":
    main()
