# app/inference.py
"""Unified inference functions for all 4 modes."""

from typing import Any

import numpy as np
import pandas as pd

from app.client import Client
from app.config import THRESHOLD
from app.encoding import encoded_plaintext_score
from app.linear_scorer import Server
from app.model import compute_manual_score, sigmoid


def plaintext_inference(
    model: Any,
    x_test: np.ndarray | pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run baseline sklearn inference.
    Accepts DataFrame or numpy array.
    """
    predictions = np.asarray(model.predict(x_test), dtype=np.int64)
    probabilities = np.asarray(model.predict_proba(x_test)[:, 1], dtype=np.float64)
    return predictions, probabilities


def manual_plaintext_inference(
    x_scaled: np.ndarray,
    w: np.ndarray,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run manual plaintext inference via linear score + sigmoid.

    Args:
        x_scaled: Scaled test feature matrix.
        w: Logistic regression weights.
        b: Logistic regression bias.

    Returns:
        Tuple ``(predictions, probabilities)``.
    """
    scores = compute_manual_score(x=x_scaled, w=w, b=b)
    probabilities = sigmoid(scores)
    predictions = (probabilities >= THRESHOLD).astype(np.int64)
    return predictions, np.asarray(probabilities, dtype=np.float64)


def encoded_plaintext_inference(
    x_scaled: np.ndarray,
    w: np.ndarray,
    b: float,
    scale: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run encoded plaintext inference (fixed-point, no encryption).

    Args:
        x_scaled: Scaled test feature matrix.
        w: Logistic regression weights.
        b: Logistic regression bias.
        scale: Fixed-point scale.
        threshold: Decision threshold.

    Returns:
        Tuple ``(predictions, probabilities)``.
    """
    scores = [
        encoded_plaintext_score(x=sample, w=w, b=b, scale=scale)
        for sample in np.asarray(x_scaled, dtype=np.float64)
    ]
    scores_array = np.asarray(scores, dtype=np.float64)
    probabilities = sigmoid(scores_array)
    predictions = (probabilities >= threshold).astype(np.int64)
    return predictions, np.asarray(probabilities, dtype=np.float64)


def phe_inference_one(
    client: Client,
    server: Server,
    x_raw: np.ndarray | pd.DataFrame,
) -> tuple[int, float]:
    """
    Execute full client-server PHE protocol for one sample.

    Accepts either a numpy array or a pandas DataFrame with one row.
    If a numpy array is given, it is wrapped into a DataFrame using
    the scaler's feature names (if available) to avoid warnings.

    Args:
        client: Client instance.
        server: Server instance.
        x_raw: One sample (array-like or DataFrame with a single row).

    Returns:
        Tuple ``(prediction, probability)``.
    """
    if isinstance(x_raw, pd.DataFrame):
        # DataFrame already has column names → no warning
        sample_df = x_raw
    else:
        # Convert numpy array to 2D and then to DataFrame
        arr = np.asarray(x_raw, dtype=np.float64).reshape(1, -1)
        scaler = client.scaler
        if hasattr(scaler, "feature_names_in_"):
            sample_df = pd.DataFrame(arr, columns=scaler.feature_names_in_)
        else:
            sample_df = pd.DataFrame(arr)

    x_scaled = client.preprocess(sample_df)
    x_int = client.encode(x_scaled)
    enc_x = client.encrypt(x_int)
    enc_score = server.compute_encrypted_score(enc_x)
    return client.decrypt_and_predict(enc_score)


def phe_inference_batch(
    client: Client,
    server: Server,
    x_raw: np.ndarray | pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Execute PHE protocol for a batch of raw samples.

    If a DataFrame is provided, iterates over rows keeping feature names
    to prevent sklearn warnings. If a numpy array is given, it is first
    converted to a DataFrame using the scaler's feature names (if available).

    Args:
        client: Client instance.
        server: Server instance.
        x_raw: Feature matrix (DataFrame or numpy array).

    Returns:
        Tuple ``(predictions, probabilities)`` for all samples.
    """
    if isinstance(x_raw, pd.DataFrame):
        df = x_raw
    else:
        arr = np.asarray(x_raw, dtype=np.float64)
        scaler = client.scaler
        if hasattr(scaler, "feature_names_in_"):
            df = pd.DataFrame(arr, columns=scaler.feature_names_in_)
        else:
            df = pd.DataFrame(arr)

    preds: list[int] = []
    probs: list[float] = []
    for i in range(len(df)):
        sample_df = df.iloc[[i]]  # сохраняет имена колонок
        pred, prob = phe_inference_one(client=client, server=server, x_raw=sample_df)
        preds.append(pred)
        probs.append(prob)

    return np.asarray(preds, dtype=np.int64), np.asarray(probs, dtype=np.float64)
