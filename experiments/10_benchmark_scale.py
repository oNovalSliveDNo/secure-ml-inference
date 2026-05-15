# experiments/10_benchmark_scale.py
"""Experiment 10: Benchmark impact of SCALE on fixed-point and PHE agreement."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from app.client import Client
from app.config import RANDOM_STATE, TEST_SIZE
from app.encoding import encode_bias, encode_weights
from app.inference import (
    encoded_plaintext_inference,
    manual_plaintext_inference,
    phe_inference_batch,
    plaintext_inference,
)
from app.model import extract_linear_params, load_model
from app.server import Server

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_PATH = Path("results/models/model.pkl")
TABLES_DIR = Path("results/tables")
SCALE_CSV_PATH = TABLES_DIR / "scale_metrics.csv"
SCALE_VALUES = [100, 1000, 10000, 100000]
PHE_SUBSET_SIZE = 15
PHE_KEY_LENGTH = 512

CSV_HEADERS = ["scale", "encoded_match_rate", "mean_abs_score_error", "phe_match_rate"]


def main() -> None:
    """Run SCALE sensitivity benchmark and write table."""
    from app.data import load_dataset, split_dataset

    model = load_model(str(MODEL_PATH))
    features, target = load_dataset()
    _, x_test, _, _ = split_dataset(
        features=features,
        target=target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    baseline_pred, _ = plaintext_inference(model=model, x_test=x_test.to_numpy())

    scaler = model.named_steps["scaler"]
    w, b = extract_linear_params(model)
    x_scaled = scaler.transform(x_test.to_numpy())

    _, manual_prob = manual_plaintext_inference(x_scaled=x_scaled, w=w, b=b)
    manual_scores = np.log(manual_prob / (1.0 - manual_prob))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int]] = []

    for scale in SCALE_VALUES:
        encoded_pred, encoded_prob = encoded_plaintext_inference(
            x_scaled=x_scaled,
            w=w,
            b=b,
            scale=scale,
            threshold=0.5,
        )
        encoded_scores = np.log(encoded_prob / (1.0 - encoded_prob))

        encoded_match_rate = float(np.mean(encoded_pred == baseline_pred))
        mean_abs_score_error = float(np.mean(np.abs(encoded_scores - manual_scores)))

        client = Client(scaler=scaler, scale=scale, key_length=PHE_KEY_LENGTH)
        server = Server(
            w_int=encode_weights(w=w, scale=scale),
            b_int=encode_bias(b=b, scale=scale),
            public_key=client.public_key,
        )

        x_subset = x_test.to_numpy()[:PHE_SUBSET_SIZE]
        encoded_subset = encoded_pred[:PHE_SUBSET_SIZE]
        phe_pred, _ = phe_inference_batch(client=client, server=server, x_raw=x_subset)
        phe_match_rate = float(np.mean(phe_pred == encoded_subset))

        rows.append(
            {
                "scale": scale,
                "encoded_match_rate": encoded_match_rate,
                "mean_abs_score_error": mean_abs_score_error,
                "phe_match_rate": phe_match_rate,
            }
        )
        logger.info(
            "SCALE=%d -> encoded_match_rate=%.6f, mean_abs_score_error=%.6f, phe_match_rate=%.6f",
            scale,
            encoded_match_rate,
            mean_abs_score_error,
            phe_match_rate,
        )

    with SCALE_CSV_PATH.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved scale metrics to %s", SCALE_CSV_PATH)


if __name__ == "__main__":
    main()
