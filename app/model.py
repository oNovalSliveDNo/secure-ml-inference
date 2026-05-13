# app/model.py
"""Baseline model training, persistence, and weight extraction."""

import numpy as np
from sklearn.pipeline import Pipeline


def train_baseline_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> Pipeline:
    """
    Train StandardScaler + LogisticRegression pipeline.

    Args:
        X_train: Training features.
        y_train: Training labels.
        random_state: Seed for reproducibility.

    Returns:
        Fitted scikit-learn Pipeline.
    """
    # TODO: create pipeline, fit, return
    raise NotImplementedError


def save_model(pipeline: Pipeline, filepath: str) -> None:
    """Save trained pipeline to disk using joblib."""
    # TODO: implement saving
    raise NotImplementedError


def load_model(filepath: str) -> Pipeline:
    """Load trained pipeline from disk."""
    # TODO: implement loading
    raise NotImplementedError


def extract_linear_params(pipeline: Pipeline) -> tuple[np.ndarray, float]:
    """
    Extract weight vector w and bias b from the logistic regression step.

    Returns:
        w: 1D array of coefficients (shape: n_features).
        b: intercept (float).
    """
    # TODO: extract coef_ and intercept_ from the pipeline's named step
    raise NotImplementedError


def compute_manual_score(
    x: np.ndarray,
    w: np.ndarray,
    b: float,
) -> np.ndarray:
    """
    Compute linear score z = X @ w + b.

    Args:
        X: Scaled feature matrix (n_samples, n_features).
        w: Weight vector (n_features).
        b: Intercept.

    Returns:
        Linear score for each sample.
    """
    # TODO: implement
    raise NotImplementedError


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute sigmoid function element-wise."""
    # TODO: implement
    raise NotImplementedError
