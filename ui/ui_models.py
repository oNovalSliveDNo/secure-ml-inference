"""Typed UI data structures for the Streamlit protocol demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    import pandas as pd

ScenarioId = Literal["classification", "regression"]
TaskType = Literal["classification", "regression"]


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


@dataclass(slots=True)
class ProtocolState:
    """State of the step-by-step protocol wizard stored in ``st.session_state``."""

    scenario_id: str
    sample_idx: int
    step: int = 1
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
    """Aggregate fidelity metrics from experiment result tables."""

    metric_name: str
    value: float
    scenario: str | None = None
    mode: str | None = None
