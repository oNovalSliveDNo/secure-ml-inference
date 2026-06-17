"""Pure, Streamlit-independent metric helpers for the UI layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

import numpy as np
import pandas as pd

from app.encoding import encoded_plaintext_score
from ui.ui_models import (
    FidelityMetrics,
    ProtocolTrace,
    RegressionErrorDistribution,
    SampleRegressionMetrics,
    StatusLevel,
    TraceValue,
)

DEFAULT_FIDELITY_TOLERANCE = 0.01


def _to_float(value: Any) -> float | None:
    """Convert scalar Python/numpy/pandas values to float when possible."""
    if value is None or value is pd.NA:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _finite_float_array(values: Iterable[Any]) -> np.ndarray:
    """Return finite float values from array-like input."""
    array = np.asarray(values, dtype=float).reshape(-1)
    return cast("np.ndarray", array[np.isfinite(array)])


def classify_sample_error_status(
    sample_abs_error: float | None, median_abs_error: float | None, p90_abs_error: float | None
) -> tuple[str, str]:
    """Classify selected regression-sample baseline error against dataset distribution."""
    if sample_abs_error is None or median_abs_error is None or p90_abs_error is None:
        return "Недостаточно данных", "off"
    if sample_abs_error <= median_abs_error:
        return "Ошибка не выше медианной", "normal"
    if sample_abs_error <= p90_abs_error:
        return "Ошибка между медианой и p90", "warning"
    return "Ошибка выше p90", "critical"


def classify_fidelity_status(delta_phe_baseline: float | None) -> tuple[str, str]:
    """Classify PHE-vs-baseline fidelity using the default UI tolerance."""
    if delta_phe_baseline is None:
        return "Недостаточно данных", "off"
    if delta_phe_baseline <= DEFAULT_FIDELITY_TOLERANCE + 1e-12:
        return "В пределах допуска", "normal"
    return "Выше допуска", "critical"


def compute_regression_error_distribution(
    y_true_all: Iterable[Any], baseline_predictions_all: Iterable[Any]
) -> RegressionErrorDistribution:
    """Compute absolute-error distribution for regression baseline predictions."""
    y_true = _finite_float_array(y_true_all)
    y_pred = _finite_float_array(baseline_predictions_all)
    sample_count = min(len(y_true), len(y_pred))
    if sample_count == 0:
        return RegressionErrorDistribution(None, None, [], 0)
    abs_errors = np.abs(y_pred[:sample_count] - y_true[:sample_count])
    return RegressionErrorDistribution(
        median_abs_error=float(np.median(abs_errors)),
        p90_abs_error=float(np.percentile(abs_errors, 90)),
        abs_errors=[float(value) for value in abs_errors],
        sample_count=int(sample_count),
    )


def compute_sample_regression_metrics(
    y_true: Any, baseline_prediction: Any, distribution: RegressionErrorDistribution
) -> SampleRegressionMetrics:
    """Compute single-sample regression metrics against a precomputed distribution."""
    true_value = _to_float(y_true)
    baseline_value = _to_float(baseline_prediction)
    abs_error = (
        abs(baseline_value - true_value)
        if baseline_value is not None and true_value is not None
        else None
    )
    relative_error = (
        abs_error / abs(true_value)
        if abs_error is not None and true_value is not None and true_value != 0.0
        else None
    )
    error_percentile = None
    if abs_error is not None and distribution.abs_errors:
        errors = np.asarray(distribution.abs_errors, dtype=float)
        error_percentile = float(100.0 * np.mean(errors <= abs_error))
    status_label, status_level = classify_sample_error_status(
        abs_error, distribution.median_abs_error, distribution.p90_abs_error
    )
    return SampleRegressionMetrics(
        true_value=true_value,
        baseline_prediction=baseline_value,
        abs_error=abs_error,
        relative_error=relative_error,
        error_percentile=error_percentile,
        median_abs_error=distribution.median_abs_error,
        p90_abs_error=distribution.p90_abs_error,
        status_label=status_label,
        status_level=cast("StatusLevel", status_level),
    )


def compute_fidelity_metrics(
    baseline_prediction: Any,
    encoded_prediction: Any,
    phe_prediction: Any,
    tolerance: float | None = DEFAULT_FIDELITY_TOLERANCE,
) -> FidelityMetrics:
    """Compute prediction deltas introduced by encoding and encrypted inference."""
    baseline = _to_float(baseline_prediction)
    encoded = _to_float(encoded_prediction)
    phe = _to_float(phe_prediction)
    tol = _to_float(tolerance)
    delta_encoded_baseline = (
        abs(encoded - baseline) if encoded is not None and baseline is not None else None
    )
    delta_phe_baseline = abs(phe - baseline) if phe is not None and baseline is not None else None
    delta_phe_encoded = abs(phe - encoded) if phe is not None and encoded is not None else None
    deltas = [
        d for d in (delta_encoded_baseline, delta_phe_baseline, delta_phe_encoded) if d is not None
    ]
    max_delta = max(deltas) if deltas else None
    margin = tol - max_delta if tol is not None and max_delta is not None else None
    status_label, status_level = classify_fidelity_status(delta_phe_baseline)
    return FidelityMetrics(
        baseline_prediction=baseline,
        encoded_prediction=encoded,
        phe_prediction=phe,
        delta_encoded_baseline=delta_encoded_baseline,
        delta_phe_baseline=delta_phe_baseline,
        delta_phe_encoded=delta_phe_encoded,
        max_delta=max_delta,
        tolerance=tol,
        margin=margin,
        status_label=status_label,
        status_level=cast("StatusLevel", status_level),
    )


def build_protocol_trace(
    *,
    result: dict[str, Any],
    scenario: dict[str, Any],
    sample: pd.Series | dict[str, Any],
    scale: int,
) -> ProtocolTrace:
    """Build protocol trace tables without rendering them in Streamlit."""
    weights = np.asarray(scenario["w"], dtype=float)
    bias = float(scenario["b"])
    x_scaled = np.asarray(result["x_scaled"], dtype=float)
    x_int = np.asarray(result["x_int"], dtype=int)
    feature_names = list(sample.index) if isinstance(sample, pd.Series) else list(sample.keys())
    encoded_score = encoded_plaintext_score(x=x_scaled, w=weights, b=bias, scale=scale)
    dot_value = float(np.dot(weights, x_scaled))
    control_z = float(dot_value + bias)
    substitutions = pd.DataFrame(
        {
            "Признак": feature_names,
            "x после масштабирования": x_scaled,
            "X = round(scale · x)": x_int,
            "w": weights,
            "w · x": weights * x_scaled,
        }
    )
    control_calculation = pd.DataFrame(
        [
            {"Величина": "Σ wᵢxᵢ", "Значение": dot_value},
            {"Величина": "b", "Значение": bias},
            {"Величина": "Открытый контрольный z", "Значение": control_z},
            {"Величина": "Открытый расчёт после кодирования", "Значение": float(encoded_score)},
            {"Величина": "Защищённый z после расшифровки", "Значение": result.get("z_secure")},
        ]
    )
    return ProtocolTrace(
        scale=int(scale),
        feature_count=len(weights),
        substitutions=substitutions,
        control_calculation=control_calculation,
        values=[
            TraceValue("Σ wᵢxᵢ", "dot(w, x_scaled)", dot_value),
            TraceValue("b", "intercept", bias),
            TraceValue("z", "dot(w, x_scaled) + b", control_z),
            TraceValue("z_encoded", "encoded_plaintext_score(...)", encoded_score),
            TraceValue("z_secure", "decrypt(server_result)", result.get("z_secure")),
        ],
    )
