# experiments/07_benchmark_feature_scaling.py
"""Experiment 07: Benchmark impact of feature count on time and size."""

from __future__ import annotations

import csv
import logging
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from app.config import FEATURE_COUNTS, KEY_LENGTH, N_BENCHMARK_RUNS, RANDOM_STATE, SCALE, TEST_SIZE
from app.crypto import decrypt_score, encrypt_vector, generate_keys, serialize_ciphertext
from app.data import load_dataset, split_dataset
from app.encoding import decode_score, encode_bias, encode_weights
from app.metrics import measure_payload_size
from app.model import extract_linear_params, load_model
from app.server import Server

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


MODEL_PATH = Path("results/models/model.pkl")
TABLES_DIR = Path("results/tables")
PLOTS_DIR = Path("results/plots")
CSV_PATH = TABLES_DIR / "feature_scaling_metrics.csv"


def _ms(start: float, end: float) -> float:
    """Convert time interval to milliseconds."""
    return (end - start) * 1000.0


def main() -> None:
    """Measure latency and payload scaling across top-k features."""
    logger.info("Feature scaling benchmark...")

    model = load_model(str(MODEL_PATH))
    features, target = load_dataset()
    _, x_test, _, _ = split_dataset(
        features=features, target=target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = model.named_steps["scaler"]
    w, b = extract_linear_params(model)
    top_indices = np.argsort(np.abs(w))[::-1]

    sample_raw = x_test.to_numpy()[0]
    x_scaled_full = scaler.transform(sample_raw.reshape(1, -1))[0]

    results: list[dict[str, float | int]] = []

    for k in FEATURE_COUNTS:
        if k <= 0 or k > len(w):
            logger.warning("Skipping invalid feature count k=%s", k)
            continue

        idx = top_indices[:k]
        w_k = w[idx]
        x_scaled_k = np.asarray(x_scaled_full[idx], dtype=np.float64)

        encryption_ms: list[float] = []
        server_ms: list[float] = []
        decryption_ms: list[float] = []
        total_ms: list[float] = []
        request_sizes: list[int] = []

        for _ in range(N_BENCHMARK_RUNS):
            t_total_start = time.perf_counter()

            public_key, private_key = generate_keys(n_length=KEY_LENGTH)
            x_int_k = np.rint(x_scaled_k * SCALE).astype(np.int64)
            w_int_k = encode_weights(w=w_k, scale=SCALE)
            b_int = encode_bias(b=b, scale=SCALE)
            server = Server(w_int=w_int_k, b_int=b_int, public_key=public_key)

            t0 = time.perf_counter()
            enc_x = encrypt_vector(public_key=public_key, x_int=[int(v) for v in x_int_k.tolist()])
            t1 = time.perf_counter()
            encryption_ms.append(_ms(t0, t1))

            request_payload = {
                "public_key_n": str(public_key.n),
                "encrypted_features": [serialize_ciphertext(value) for value in enc_x],
                "scale": SCALE,
                "feature_count": k,
            }
            request_sizes.append(measure_payload_size(request_payload))

            t0 = time.perf_counter()
            enc_score = server.compute_encrypted_score(enc_x)
            t1 = time.perf_counter()
            server_ms.append(_ms(t0, t1))

            t0 = time.perf_counter()
            score_int = decrypt_score(private_key=private_key, encrypted_score=enc_score)
            _ = decode_score(score_int=score_int, scale=SCALE)
            t1 = time.perf_counter()
            decryption_ms.append(_ms(t0, t1))

            t_total_end = time.perf_counter()
            total_ms.append(_ms(t_total_start, t_total_end))

        results.append(
            {
                "feature_count": int(k),
                "encryption_mean_ms": float(np.mean(encryption_ms)),
                "server_mean_ms": float(np.mean(server_ms)),
                "decryption_mean_ms": float(np.mean(decryption_ms)),
                "total_mean_ms": float(np.mean(total_ms)),
                "request_size_mean_bytes": float(np.mean(request_sizes)),
            }
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "feature_count",
                "encryption_mean_ms",
                "server_mean_ms",
                "decryption_mean_ms",
                "total_mean_ms",
                "request_size_mean_bytes",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    counts = [int(row["feature_count"]) for row in results]
    total_vals = [float(row["total_mean_ms"]) for row in results]
    req_vals = [float(row["request_size_mean_bytes"]) for row in results]
    server_vals = [float(row["server_mean_ms"]) for row in results]

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(counts, total_vals, marker="o")
    plt.xlabel("Feature count (k)")
    plt.ylabel("Total time (ms)")
    plt.title("Feature count vs total time")
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / "feature_count_vs_total_time.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(counts, req_vals, marker="o")
    plt.xlabel("Feature count (k)")
    plt.ylabel("Encrypted request size (bytes)")
    plt.title("Feature count vs request size")
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / "feature_count_vs_request_size.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(counts, server_vals, marker="o")
    plt.xlabel("Feature count (k)")
    plt.ylabel("Server compute time (ms)")
    plt.title("Feature count vs server time")
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / "feature_count_vs_server_time.png", bbox_inches="tight")
    plt.close()

    logger.info("Saved feature scaling metrics to %s", CSV_PATH)
    logger.info("Saved plots to %s", PLOTS_DIR)
    logger.info("Feature scaling benchmark completed.")


if __name__ == "__main__":
    main()
