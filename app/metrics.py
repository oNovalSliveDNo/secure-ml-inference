# app/metrics.py
"""Utility functions for classification metrics, timing, and payload size."""

import json
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute standard binary classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for class 1

    Returns:
        Dictionary with accuracy, precision, recall, f1, and optional roc_auc.
    """
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def compare_predictions(
    pred1: np.ndarray,
    pred2: np.ndarray,
    score1: np.ndarray | None = None,
    score2: np.ndarray | None = None,
) -> dict[str, float]:
    """Compare two prediction outputs.

    Args:
        pred1: First prediction array.
        pred2: Second prediction array.
        score1: Optional first probability/score array.
        score2: Optional second probability/score array.

    Returns:
        Dictionary with ``match_rate`` and optional ``mean_abs_score_error``.
    """
    pred1_array = np.asarray(pred1)
    pred2_array = np.asarray(pred2)
    result: dict[str, float] = {"match_rate": float(np.mean(pred1_array == pred2_array))}
    if score1 is not None and score2 is not None:
        score1_array = np.asarray(score1, dtype=np.float64)
        score2_array = np.asarray(score2, dtype=np.float64)
        result["mean_abs_score_error"] = float(np.mean(np.abs(score1_array - score2_array)))
    return result


def timing_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure execution time in milliseconds.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function returning ``(original_result, elapsed_ms)``.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, float]:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return result, float(elapsed_ms)

    return wrapper


def measure_payload_size(obj: Any) -> int:
    """
    Estimate payload size in bytes for JSON-serialized object.

    Args:
        obj: Python object serializable to JSON.

    Returns:
        UTF-8 byte length of the serialized JSON string.
    """
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return len(payload.encode("utf-8"))
