# experiments/13_run_phe_regression.py
"""Experiment 13: Evaluate encoded and PHE regression inference quality."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.client import Client
from app.config import KEY_LENGTH, SCALE
from app.encoding import decode_score, encode_bias, encode_weights
from app.linear_scorer import EncryptedLinearScorer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODELS_DIR = Path("results/models")
MODEL_PATH = MODELS_DIR / "regression_model.pkl"
SCALER_PATH = MODELS_DIR / "regression_scaler.pkl"
TABLES_DIR = Path("results/tables")
QUALITY_CSV_PATH = TABLES_DIR / "regression_quality_metrics.csv"
CSV_HEADERS = [
    "mode",
    "mae",
    "mse",
    "rmse",
    "r2",
    "diff_mae",
    "diff_mse",
    "diff_rmse",
    "diff_r2",
    "match_rate_tol_1e_2",
]


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": mse**0.5,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _append_quality_row(row: dict[str, float | str]) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = QUALITY_CSV_PATH.exists()
    with QUALITY_CSV_PATH.open("a", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _build_row(
    mode: str,
    metrics: dict[str, float],
    ref_pred: np.ndarray,
    cur_pred: np.ndarray,
) -> dict[str, float | str]:
    diff = np.asarray(cur_pred, dtype=np.float64) - np.asarray(ref_pred, dtype=np.float64)
    diff_mse = float(np.mean(diff**2))
    return {
        "mode": mode,
        "mae": metrics["mae"],
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "diff_mae": float(np.mean(np.abs(diff))),
        "diff_mse": diff_mse,
        "diff_rmse": diff_mse**0.5,
        "diff_r2": float(r2_score(ref_pred, cur_pred)),
        "match_rate_tol_1e_2": float(np.mean(np.abs(diff) <= 1e-2)),
    }


def main() -> None:
    logger.info("Starting regression encoded/PHE evaluation...")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    dataset = load_diabetes(as_frame=True)
    features = dataset.data
    target = dataset.target
    _, x_test, _, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    y_true = y_test.to_numpy(dtype=np.float64)
    y_pred_baseline = np.asarray(model.predict(x_test), dtype=np.float64)

    regressor = model.named_steps["regressor"]
    w = np.asarray(regressor.coef_, dtype=np.float64)
    b = float(regressor.intercept_)

    w_int = encode_weights(w=w, scale=SCALE)
    b_int = encode_bias(b=b, scale=SCALE)

    encoded_scores = []
    phe_scores = []

    client = Client(scaler=scaler, scale=SCALE, key_length=KEY_LENGTH)
    server = EncryptedLinearScorer(w_int=w_int, b_int=b_int, public_key=client.public_key)

    for sample_raw in np.asarray(x_test, dtype=np.float64):
        x_raw_2d = sample_raw.reshape(1, -1)
        x_scaled_one = client.preprocess(x_raw_2d)
        x_int = client.encode(x_scaled_one)

        encoded_score = decode_score(int(np.dot(x_int, w_int) + b_int), scale=SCALE)
        encoded_scores.append(encoded_score)

        enc_x = client.encrypt(x_int)
        enc_score = server.compute_encrypted_score(enc_x)
        phe_score_int = client.private_key.decrypt(enc_score)
        phe_scores.append(decode_score(score_int=phe_score_int, scale=SCALE))

    y_pred_encoded = np.asarray(encoded_scores, dtype=np.float64)
    y_pred_phe = np.asarray(phe_scores, dtype=np.float64)

    encoded_metrics = _regression_metrics(y_true=y_true, y_pred=y_pred_encoded)
    phe_metrics = _regression_metrics(y_true=y_true, y_pred=y_pred_phe)

    logger.info("Encoded metrics: %s", encoded_metrics)
    logger.info("PHE metrics: %s", phe_metrics)

    _append_quality_row(
        _build_row("Encoded plaintext", encoded_metrics, y_pred_baseline, y_pred_encoded)
    )
    _append_quality_row(_build_row("PHE inference", phe_metrics, y_pred_encoded, y_pred_phe))

    logger.info("Saved regression quality rows to %s", QUALITY_CSV_PATH)


if __name__ == "__main__":
    main()
