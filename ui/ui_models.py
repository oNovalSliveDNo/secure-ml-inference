"""Typed UI data structures for the Streamlit protocol demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    import pandas as pd

ScenarioId = Literal["classification", "regression"]
TaskType = Literal["classification", "regression"]
StatusLevel = Literal["off", "normal", "warning", "critical"]


class TransportMetadata(TypedDict, total=False):
    """Metadata sent with or received from the encrypted inference endpoint."""

    public_key_n: str
    encrypted_features: list[str]
    scale: int
    scenario_id: str
    feature_count: int
    status_code: int
    server_compute_ms: float | None
    http_elapsed_ms: float
    plaintext_bytes: int
    encrypted_bytes: int
    overhead_ratio: float


@dataclass(slots=True)
class ProtocolState:
    """State of the step-by-step protocol wizard stored in ``st.session_state``."""

    scenario_id: str
    sample_idx: int
    step: int = 0
    result: dict[str, Any] = field(default_factory=dict)

    def to_session_dict(self) -> dict[str, Any]:
        """Return a Streamlit-session friendly representation."""
        return {
            "scenario_id": self.scenario_id,
            "sample_idx": self.sample_idx,
            "step": self.step,
            "result": self.result,
        }


@dataclass(slots=True)
class TraceValue:
    """Single value shown in the mathematical trace of the protocol."""

    name: str
    formula: str
    value: str | int | float | None


@dataclass(slots=True)
class ProtocolTrace:
    """Human-readable explanation of plaintext and encoded protocol calculations."""

    scale: int
    feature_count: int
    substitutions: pd.DataFrame
    control_calculation: pd.DataFrame
    values: list[TraceValue] = field(default_factory=list)


@dataclass(slots=True)
class RegressionErrorDistribution:
    """Baseline absolute-error distribution for a regression dataset."""

    median_abs_error: float | None
    p90_abs_error: float | None
    abs_errors: list[float] = field(default_factory=list)
    sample_count: int = 0


@dataclass(slots=True)
class SampleRegressionMetrics:
    """Regression quality metrics for one selected sample."""

    true_value: float | None
    baseline_prediction: float | None
    abs_error: float | None
    relative_error: float | None
    error_percentile: float | None
    median_abs_error: float | None
    p90_abs_error: float | None
    status_label: str
    status_level: StatusLevel


@dataclass(slots=True)
class SampleMetrics:
    """Metrics calculated for a single selected sample."""

    plaintext_bytes: int
    encrypted_bytes: int
    overhead_ratio: float
    baseline_value: float | int | None = None
    encoded_value: float | int | None = None
    secure_value: float | int | None = None
    true_value: float | int | None = None


@dataclass(slots=True)
class FidelityMetrics:
    """Sample-level parity metrics for baseline, encoded, and PHE predictions."""

    baseline_prediction: float | None
    encoded_prediction: float | None
    phe_prediction: float | None
    delta_encoded_baseline: float | None
    delta_phe_baseline: float | None
    delta_phe_encoded: float | None
    max_delta: float | None
    tolerance: float | None
    margin: float | None
    status_label: str
    status_level: StatusLevel
    scenario: str | None = None
    mode: str | None = None
