# experiments/05_benchmark_latency.py
"""Experiment 05: Benchmark latency components of PHE inference."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import numpy as np

from app.config import KEY_LENGTH, N_BENCHMARK_RUNS, RANDOM_STATE, SCALE, TEST_SIZE, THRESHOLD
from app.crypto import decrypt_score, encrypt_vector, generate_keys
from app.data import load_dataset, split_dataset
from app.encoding import decode_score, encode_bias, encode_vector, encode_weights
from app.model import extract_linear_params, load_model
from app.server import Server

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_PATH = Path("results/models/model.pkl")
TABLES_DIR = Path("results/tables")
LATENCY_CSV_PATH = TABLES_DIR / "latency_metrics.csv"

STAGES = [
    "preprocessing",
    "encoding",
    "key_generation",
    "encryption",
    "server_compute",
    "decryption",
    "sigmoid_threshold",
    "total",
]


def _to_ms(start: float, end: float) -> float:
    """Convert perf_counter interval to milliseconds."""
    return (end - start) * 1000.0


def _summary(values: list[float]) -> tuple[float, float, float]:
    """Return mean, std, median for a list of milliseconds."""
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array)), float(np.median(array))


def _write_latency_csv(stage_to_values: dict[str, list[float]]) -> None:
    """Write latency summary rows to CSV file.

    Args:
        stage_to_values: Mapping of stage name to measured milliseconds.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with LATENCY_CSV_PATH.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["stage", "mean_ms", "std_ms", "median_ms"])
        writer.writeheader()
        for stage in STAGES:
            mean_ms, std_ms, median_ms = _summary(stage_to_values[stage])
            writer.writerow(
                {
                    "stage": stage,
                    "mean_ms": mean_ms,
                    "std_ms": std_ms,
                    "median_ms": median_ms,
                }
            )


def main() -> None:
    """Benchmark end-to-end and component latency of PHE inference."""
    logger.info("Benchmarking latency...")

    model = load_model(str(MODEL_PATH))
    features, target = load_dataset()
    _, x_test, _, _ = split_dataset(
        features=features, target=target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = model.named_steps["scaler"]
    w, b = extract_linear_params(model)
    w_int = encode_weights(w=w, scale=SCALE)
    b_int = encode_bias(b=b, scale=SCALE)
    sample = x_test.to_numpy()[0]

    measurements: dict[str, list[float]] = {stage: [] for stage in STAGES}

    for _ in range(N_BENCHMARK_RUNS):
        t_total_start = time.perf_counter()

        t0 = time.perf_counter()
        x_scaled = scaler.transform(sample.reshape(1, -1))[0]
        t1 = time.perf_counter()
        measurements["preprocessing"].append(_to_ms(t0, t1))

        t0 = time.perf_counter()
        x_int = encode_vector(x=x_scaled, scale=SCALE)
        t1 = time.perf_counter()
        measurements["encoding"].append(_to_ms(t0, t1))

        t0 = time.perf_counter()
        public_key, private_key = generate_keys(n_length=KEY_LENGTH)
        t1 = time.perf_counter()
        measurements["key_generation"].append(_to_ms(t0, t1))

        server = Server(w_int=w_int, b_int=b_int, public_key=public_key)

        t0 = time.perf_counter()
        enc_x = encrypt_vector(public_key=public_key, x_int=[int(v) for v in x_int.tolist()])
        t1 = time.perf_counter()
        measurements["encryption"].append(_to_ms(t0, t1))

        t0 = time.perf_counter()
        enc_score = server.compute_encrypted_score(enc_x)
        t1 = time.perf_counter()
        measurements["server_compute"].append(_to_ms(t0, t1))

        t0 = time.perf_counter()
        score_int = decrypt_score(private_key=private_key, encrypted_score=enc_score)
        t1 = time.perf_counter()
        measurements["decryption"].append(_to_ms(t0, t1))

        t0 = time.perf_counter()
        z = decode_score(score_int=score_int, scale=SCALE)
        probability = float(1.0 / (1.0 + np.exp(-z)))
        _ = int(probability >= THRESHOLD)
        t1 = time.perf_counter()
        measurements["sigmoid_threshold"].append(_to_ms(t0, t1))

        t_total_end = time.perf_counter()
        measurements["total"].append(_to_ms(t_total_start, t_total_end))

    _write_latency_csv(stage_to_values=measurements)
    logger.info("Saved latency metrics to %s", LATENCY_CSV_PATH)
    logger.info("Latency benchmark completed.")


if __name__ == "__main__":
    main()
