# experiments/11_benchmark_api_roundtrip.py
"""Experiment 11: Benchmark end-to-end API latency via real HTTP requests."""

from __future__ import annotations

import csv
import logging
import os
import time
from pathlib import Path

import numpy as np
import requests

from app.client import Client
from app.config import KEY_LENGTH, N_BENCHMARK_RUNS, RANDOM_STATE, SCALE, TEST_SIZE, THRESHOLD
from app.crypto import deserialize_ciphertext, serialize_ciphertext
from app.data import load_dataset, split_dataset
from app.model import load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_PATH = Path("results/models/model.pkl")
TABLES_DIR = Path("results/tables")
API_ROUNDTRIP_CSV_PATH = TABLES_DIR / "api_roundtrip_metrics.csv"
API_URL = os.getenv("API_URL", "http://localhost:8000")
INFER_ENDPOINT = "/infer/encrypted"

CSV_HEADERS = [
    "client_prepare_ms",
    "http_roundtrip_ms",
    "server_compute_ms",
    "decrypt_predict_ms",
    "total_ms",
]


def _to_ms(start: float, end: float) -> float:
    """Convert perf_counter interval to milliseconds."""
    return (end - start) * 1000.0


def _mean(values: list[float]) -> float:
    """Return mean value for a list of milliseconds."""
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _write_roundtrip_csv(metrics: dict[str, list[float]]) -> None:
    """Persist averaged roundtrip benchmark metrics to CSV."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    row = {header: _mean(metrics[header]) for header in CSV_HEADERS}
    with API_ROUNDTRIP_CSV_PATH.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """Run client-server benchmark using real HTTP requests to FastAPI."""
    logger.info("Benchmarking API roundtrip latency via HTTP at %s", API_URL)

    model = load_model(str(MODEL_PATH))
    features, target = load_dataset()
    _, x_test, _, _ = split_dataset(
        features=features,
        target=target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    sample = x_test.to_numpy()[0]
    scaler = model.named_steps["scaler"]

    metrics: dict[str, list[float]] = {header: [] for header in CSV_HEADERS}
    infer_url = f"{API_URL.rstrip('/')}" + INFER_ENDPOINT

    for _ in range(N_BENCHMARK_RUNS):
        client = Client(scaler=scaler, scale=SCALE, key_length=KEY_LENGTH)
        t_total_start = time.perf_counter()

        t0 = time.perf_counter()
        x_scaled = client.preprocess(sample.reshape(1, -1))
        x_int = client.encode(x_scaled)
        enc_x = client.encrypt(x_int)
        payload = {
            "public_key_n": str(client.public_key.n),
            "encrypted_features": [serialize_ciphertext(value) for value in enc_x],
            "scale": SCALE,
            "feature_count": len(enc_x),
        }
        t1 = time.perf_counter()
        metrics["client_prepare_ms"].append(_to_ms(t0, t1))

        t0 = time.perf_counter()
        response = requests.post(infer_url, json=payload, timeout=30)
        response.raise_for_status()
        t1 = time.perf_counter()
        metrics["http_roundtrip_ms"].append(_to_ms(t0, t1))

        response_payload = response.json()
        metrics["server_compute_ms"].append(float(response_payload["server_compute_ms"]))

        encrypted_score = deserialize_ciphertext(
            public_key=client.public_key,
            ciphertext_str=response_payload["encrypted_score"],
        )

        t0 = time.perf_counter()
        _, probability = client.decrypt_and_predict(encrypted_score=encrypted_score)
        _ = int(probability >= THRESHOLD)
        t1 = time.perf_counter()
        metrics["decrypt_predict_ms"].append(_to_ms(t0, t1))

        t_total_end = time.perf_counter()
        metrics["total_ms"].append(_to_ms(t_total_start, t_total_end))

    _write_roundtrip_csv(metrics=metrics)
    logger.info("Saved API roundtrip metrics to %s", API_ROUNDTRIP_CSV_PATH)


if __name__ == "__main__":
    main()
