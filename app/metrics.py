# app/metrics.py
"""Utility functions for classification metrics, timing, and payload size."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import time
from typing import Dict


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> Dict[str, float]:
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
) -> Dict[str, float]:
    """
    Compare two prediction arrays.

    Returns:
        Dictionary with 'match_rate' and 'mean_abs_score_error' (if scores provided).
    """
    # TODO: compute match rate
    raise NotImplementedError


def timing_decorator(func):
    """Decorator to measure execution time."""
    # TODO: implement wrapper using time.perf_counter
    raise NotImplementedError


def measure_payload_size(obj) -> int:
    """
    Estimate size in bytes of an object when serialized as JSON.

    Args:
        obj: A Python object (list, dict, etc.)

    Returns:
        Size in bytes.
    """
    # TODO: use json.dumps and sys.getsizeof
    raise NotImplementedError
