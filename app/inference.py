# app/inference.py
"""Unified inference functions for all 4 modes."""

from typing import Any

import numpy as np

from app.client import Client
from app.config import THRESHOLD
from app.encoding import encoded_plaintext_score
from app.linear_scorer import Server
from app.model import compute_manual_score, sigmoid


def plaintext_inference(
    model: Any,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run baseline sklearn inference.

    Args:
        model: Fitted sklearn pipeline/model with ``predict`` and ``predict_proba``.
        x_test: Test feature matrix.

    Returns:
        Tuple ``(predictions, probabilities)`` where probabilities are for class 1.
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
    x_raw: np.ndarray,
) -> tuple[int, float]:
    """
    Execute full client-server PHE protocol for one sample.

    Args:
        client: Client instance.
        server: Server instance.
        x_raw: Raw feature vector for one sample.

    Returns:
        Tuple ``(prediction, probability)``.
    """
    x_raw_2d = np.asarray(x_raw, dtype=np.float64).reshape(1, -1)
    x_scaled = client.preprocess(x_raw_2d)
    x_int = client.encode(x_scaled)
    enc_x = client.encrypt(x_int)
    enc_score = server.compute_encrypted_score(enc_x)
    return client.decrypt_and_predict(enc_score)


def phe_inference_batch(
    client: Client,
    server: Server,
    x_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Execute PHE protocol for a batch of raw samples.

    Args:
        client: Client instance.
        server: Server instance.
        x_raw: Raw feature matrix.

    Returns:
        Tuple ``(predictions, probabilities)`` for all samples.
    """
    preds: list[int] = []
    probs: list[float] = []
    for sample in np.asarray(x_raw, dtype=np.float64):
        pred, prob = phe_inference_one(client=client, server=server, x_raw=sample)
        preds.append(pred)
        probs.append(prob)
    return np.asarray(preds, dtype=np.int64), np.asarray(probs, dtype=np.float64)
