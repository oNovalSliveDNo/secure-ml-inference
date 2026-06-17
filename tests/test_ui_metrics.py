"""Tests for UI metric calculations and status thresholds."""

from __future__ import annotations

import pytest

from ui.metrics_helpers import (
    compute_fidelity_metrics,
    compute_regression_error_distribution,
    compute_sample_regression_metrics,
)


def test_regression_sample_abs_error_and_distribution_thresholds() -> None:
    """sample_abs_error is abs(y_true - baseline); median/p90 drive green/red states."""
    distribution = compute_regression_error_distribution(
        y_true_all=[100.0, 100.0, 100.0, 100.0, 100.0],
        baseline_predictions_all=[100.0, 101.0, 102.0, 103.0, 110.0],
    )

    green_sample = compute_sample_regression_metrics(
        y_true=100.0,
        baseline_prediction=101.0,
        distribution=distribution,
    )
    red_sample = compute_sample_regression_metrics(
        y_true=100.0,
        baseline_prediction=111.0,
        distribution=distribution,
    )

    assert green_sample.abs_error == pytest.approx(abs(100.0 - 101.0))
    assert green_sample.median_abs_error == pytest.approx(2.0)
    assert green_sample.p90_abs_error == pytest.approx(7.2)
    assert green_sample.status_level == "normal"  # rendered as the green/OK status in the UI

    assert red_sample.abs_error == pytest.approx(abs(100.0 - 111.0))
    assert red_sample.status_level == "critical"  # rendered as the red/critical status in the UI


@pytest.mark.parametrize(
    ("baseline", "encoded", "phe", "expected_phe_baseline", "expected_phe_encoded"),
    [
        (0.80, 0.805, 0.81, 0.01, 0.005),
        (0.80, 0.86, 0.851, 0.051, 0.009),
    ],
)
def test_fidelity_deltas_are_absolute_differences(
    baseline: float,
    encoded: float,
    phe: float,
    expected_phe_baseline: float,
    expected_phe_encoded: float,
) -> None:
    """PHE-vs-baseline and PHE-vs-encoded deltas are absolute differences."""
    metrics = compute_fidelity_metrics(
        baseline_prediction=baseline,
        encoded_prediction=encoded,
        phe_prediction=phe,
    )

    assert metrics.delta_phe_baseline == pytest.approx(expected_phe_baseline)
    assert metrics.delta_phe_encoded == pytest.approx(expected_phe_encoded)


def test_fidelity_delta_thresholds_green_and_red() -> None:
    """PHE delta <= 0.01 is green/normal; PHE delta > 0.05 is red/critical."""
    green_metrics = compute_fidelity_metrics(
        baseline_prediction=1.0,
        encoded_prediction=1.005,
        phe_prediction=1.01,
    )
    red_metrics = compute_fidelity_metrics(
        baseline_prediction=1.0,
        encoded_prediction=1.02,
        phe_prediction=1.051,
    )

    assert green_metrics.delta_phe_baseline == pytest.approx(0.01)
    assert green_metrics.status_level == "normal"  # green/OK in rendered status

    assert red_metrics.delta_phe_baseline == pytest.approx(0.051)
    assert red_metrics.status_level == "critical"  # red/critical in rendered status
