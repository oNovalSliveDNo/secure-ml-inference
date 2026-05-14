# app/model.py
"""Baseline model training, persistence, and weight extraction."""

from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_baseline_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> Pipeline:
    """
    Train StandardScaler + LogisticRegression pipeline.

    Args:
        x_train: Training features.
        y_train: Training labels.
        random_state: Seed for reproducibility.

    Returns:
        Fitted scikit-learn Pipeline.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline


def save_model(pipeline: Pipeline, filepath: str) -> None:
    """Save trained pipeline to disk using joblib."""
    if pipeline is None:
        raise ValueError("pipeline must not be None")
    joblib.dump(pipeline, filepath)


def load_model(filepath: str) -> Pipeline:
    """Load trained pipeline from disk."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)


def extract_linear_params(pipeline: Pipeline) -> tuple[np.ndarray, float]:
    """
    Extract weight vector w and bias b from the logistic regression step.

    Args:
        pipeline: Fitted baseline pipeline.

    Returns:
        w: 1D array of coefficients (shape: n_features).
        b: Intercept value.

    Raises:
        ValueError: If pipeline is None or missing the logistic regression step.
    """
    if pipeline is None:
        raise ValueError("pipeline must not be None")

    model = pipeline.named_steps.get("logreg")
    if model is None:
        raise ValueError("pipeline must contain a 'logreg' step")

    w = model.coef_.ravel()
    b = float(model.intercept_.ravel()[0])
    return w, b


def compute_manual_score(
    x: np.ndarray,
    w: np.ndarray,
    b: float,
) -> NDArray[np.float64]:
    """
    Compute linear score z = X @ w + b.

    Args:
        x: Scaled feature matrix (n_samples, n_features).
        w: Weight vector (n_features).
        b: Intercept.

    Returns:
        Linear score for each sample.
    """
    return np.asarray(x @ w + b)


def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute sigmoid function element-wise."""
    return np.asarray(1.0 / (1.0 + np.exp(-z)))
