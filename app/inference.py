# app/inference.py
"""Unified inference functions for all 4 modes."""

import numpy as np
from typing import Tuple
from app.model import compute_manual_score, sigmoid
from app.encoding import encoded_plaintext_score


def plaintext_inference(model, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard sklearn inference.

    Returns:
        predictions (0/1), probabilities (class 1).
    """
    # TODO: use model.predict and predict_proba
    raise NotImplementedError


def manual_plaintext_inference(
    X_scaled: np.ndarray,
    w: np.ndarray,
    b: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inference via manual weight multiplication.

    Returns:
        predictions, probabilities.
    """
    # TODO: z = X @ w + b, p = sigmoid(z), class = (p >= threshold)
    raise NotImplementedError


def encoded_plaintext_inference(
    X_scaled: np.ndarray,
    w: np.ndarray,
    b: float,
    scale: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fixed-point encoded inference without encryption.

    Returns:
        predictions, probabilities.
    """
    # TODO: loop over samples, use encoded_plaintext_score
    raise NotImplementedError


def phe_inference_one(client, server, x_raw: np.ndarray) -> Tuple[int, float]:
    """
    Execute full PHE protocol for a single sample.

    Args:
        client: Client instance.
        server: Server instance.
        x_raw: Raw (unscaled) feature vector for one sample.

    Returns:
        (prediction, probability).
    """
    # TODO: preprocess, encode, encrypt, send to server, decrypt, predict
    raise NotImplementedError


def phe_inference_batch(
    client, server, X_raw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Execute PHE protocol for multiple samples.

    Returns:
        predictions array, probabilities array.
    """
    # TODO: iterate over samples and collect results
    raise NotImplementedError
