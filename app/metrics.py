# app/metrics.py
"""Utility functions for classification metrics, timing, and payload size."""

from collections.abc import Callable
from typing import Any

import numpy as np


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute standard classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for class 1 (optional, for ROC-AUC).

    Returns:
        Dictionary with accuracy, precision, recall, f1, roc_auc (if y_prob provided).
    """
    # TODO: implement
    raise NotImplementedError


def compare_predictions(
    pred1: np.ndarray,
    pred2: np.ndarray,
) -> dict[str, float]:
    """
    Compare two prediction arrays.

    Returns:
        Dictionary with 'match_rate' and 'mean_abs_score_error' (if scores provided).
    """
    # TODO: compute match rate
    raise NotImplementedError


def timing_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure execution time."""
    # TODO: implement wrapper using time.perf_counter
    raise NotImplementedError


def measure_payload_size(obj: Any) -> int:
    """
    Estimate size in bytes of an object when serialized as JSON.

    Args:
        obj: A Python object (list, dict, etc.)

    Returns:
        Size in bytes.
    """
    # TODO: use json.dumps and sys.getsizeof
    raise NotImplementedError
