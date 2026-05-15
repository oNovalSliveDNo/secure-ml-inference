# experiments/08_benchmark_datasets.py
"""Experiment 08: Benchmark baseline, encoded, and PHE inference across datasets."""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from app.client import Client
from app.config import KEY_LENGTH, RANDOM_STATE, SCALE, TEST_SIZE
from app.encoding import encode_bias, encode_weights
from app.inference import encoded_plaintext_inference, phe_inference_batch, plaintext_inference
from app.model import extract_linear_params, train_baseline_model
from app.server import Server

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TABLES_DIR = Path("results/tables")
DATASET_BENCHMARK_CSV_PATH = TABLES_DIR / "dataset_benchmark_metrics.csv"
PHE_SUBSET_SIZE = 25
KEY_LENGTH_FOR_BENCHMARK = KEY_LENGTH

CSV_HEADERS = [
    "dataset",
    "key_length",
    "n_samples",
    "n_features",
    "baseline_accuracy",
    "baseline_f1",
    "encoded_match_rate",
    "phe_match_rate",
    "mean_phe_latency_ms",
    "encrypted_request_size_bytes",
]


def _load_dataset_scenarios() -> list[tuple[str, pd.DataFrame, pd.Series]]:
    """Build dataset scenarios for benchmarking.

    Returns:
        List of tuples: (scenario_name, features, labels).
    """
    scenarios: list[tuple[str, pd.DataFrame, pd.Series]] = []

    bc = load_breast_cancer(as_frame=True)
    scenarios.append(("breast_cancer", bc.data, bc.target.astype(np.int64)))

    iris = load_iris(as_frame=True)
    iris_df = iris.frame
    iris_binary = iris_df[iris_df["target"].isin([0, 1])].copy()
    scenarios.append(
        (
            "iris_binary_0_1",
            iris_binary[iris.feature_names],
            iris_binary["target"].astype(np.int64),
        )
    )

    wine = load_wine(as_frame=True)
    wine_x = wine.data.copy()
    wine_y = wine.target.copy().replace({1: 1, 2: 1}).astype(np.int64)
    scenarios.append(("wine_binary_0_vs_1_2", wine_x, wine_y))

    x_50, y_50 = make_classification(
        n_samples=200,
        n_features=50,
        n_informative=20,
        n_redundant=10,
        n_repeated=0,
        n_classes=2,
        random_state=42,
    )
    scenarios.append(
        (
            "synthetic_50f_200s",
            pd.DataFrame(x_50),
            pd.Series(y_50, dtype=np.int64),
        )
    )

    x_100, y_100 = make_classification(
        n_samples=200,
        n_features=100,
        n_informative=40,
        n_redundant=20,
        n_repeated=0,
        n_classes=2,
        random_state=42,
    )
    scenarios.append(
        (
            "synthetic_100f_200s",
            pd.DataFrame(x_100),
            pd.Series(y_100, dtype=np.int64),
        )
    )

    return scenarios


def _estimate_request_size_bytes(client: Client, sample_raw: np.ndarray) -> int:
    """Estimate encrypted request payload size as serialized JSON length.

    Args:
        client: Client used for preprocessing/encoding/encryption.
        sample_raw: One raw sample.

    Returns:
        Serialized encrypted request size in bytes.
    """
    x_scaled = client.preprocess(sample_raw.reshape(1, -1))
    x_int = client.encode(x_scaled)
    encrypted_vector = client.encrypt(x_int)
    payload = {
        "encrypted_features": [str(value.ciphertext()) for value in encrypted_vector],
        "public_key_n": str(client.public_key.n),
        "scale": client.scale,
    }
    return len(json.dumps(payload).encode("utf-8"))


def _benchmark_one_dataset(
    dataset_name: str,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float | int | str]:
    """Run full benchmark pipeline for one dataset scenario.

    Args:
        dataset_name: Scenario name.
        features: Feature matrix.
        target: Binary labels.

    Returns:
        Metrics dictionary for CSV output.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = train_baseline_model(
        x_train=x_train.to_numpy(),
        y_train=y_train.to_numpy(),
        random_state=RANDOM_STATE,
    )

    baseline_pred, _ = plaintext_inference(model=model, x_test=x_test.to_numpy())
    baseline_accuracy = float(accuracy_score(y_test.to_numpy(), baseline_pred))
    baseline_f1 = float(f1_score(y_test.to_numpy(), baseline_pred))

    scaler = model.named_steps["scaler"]
    w, b = extract_linear_params(model)
    x_scaled = scaler.transform(x_test.to_numpy())

    encoded_pred, _ = encoded_plaintext_inference(
        x_scaled=x_scaled,
        w=w,
        b=b,
        scale=SCALE,
        threshold=0.5,
    )
    encoded_match_rate = float(np.mean(encoded_pred == baseline_pred))

    client = Client(scaler=scaler, scale=SCALE, key_length=KEY_LENGTH_FOR_BENCHMARK)
    server = Server(
        w_int=encode_weights(w=w, scale=SCALE),
        b_int=encode_bias(b=b, scale=SCALE),
        public_key=client.public_key,
    )

    subset_size = min(PHE_SUBSET_SIZE, len(x_test))
    x_test_subset = x_test.to_numpy()[:subset_size]
    encoded_subset = encoded_pred[:subset_size]

    latency_values_ms: list[float] = []
    phe_predictions: list[int] = []
    for sample in x_test_subset:
        start = time.perf_counter()
        pred, _ = phe_inference_batch(client=client, server=server, x_raw=sample.reshape(1, -1))
        end = time.perf_counter()
        latency_values_ms.append((end - start) * 1000.0)
        phe_predictions.append(int(pred[0]))

    phe_pred_array = np.asarray(phe_predictions, dtype=np.int64)
    phe_match_rate = float(np.mean(phe_pred_array == encoded_subset))
    mean_phe_latency_ms = float(np.mean(latency_values_ms)) if latency_values_ms else 0.0

    encrypted_request_size_bytes = _estimate_request_size_bytes(
        client=client,
        sample_raw=x_test_subset[0],
    )

    return {
        "dataset": dataset_name,
        "key_length": KEY_LENGTH_FOR_BENCHMARK,
        "n_samples": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "baseline_accuracy": baseline_accuracy,
        "baseline_f1": baseline_f1,
        "encoded_match_rate": encoded_match_rate,
        "phe_match_rate": phe_match_rate,
        "mean_phe_latency_ms": mean_phe_latency_ms,
        "encrypted_request_size_bytes": encrypted_request_size_bytes,
    }


def main() -> None:
    """Run benchmark across all configured datasets and save a CSV table."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []

    for dataset_name, features, target in _load_dataset_scenarios():
        logger.info("Benchmarking dataset: %s", dataset_name)
        rows.append(
            _benchmark_one_dataset(dataset_name=dataset_name, features=features, target=target)
        )

    with DATASET_BENCHMARK_CSV_PATH.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved dataset benchmark table to %s", DATASET_BENCHMARK_CSV_PATH)


if __name__ == "__main__":
    main()
