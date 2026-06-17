"""Readable model labels for UI views."""

from __future__ import annotations

MODEL_LABELS: dict[str, str] = {
    "classification": "логистическая регрессия",
    "regression": "гребневая регрессия Ridge",
}

MODEL_TECHNICAL_NAMES: dict[str, str] = {
    "classification": "LogisticRegression",
    "regression": "Ridge",
}


def get_model_label(scenario_id: str | None) -> str:
    """Return a Russian readable model label for a scenario."""
    return MODEL_LABELS.get(scenario_id or "", MODEL_LABELS["regression"])


def get_model_technical_name(scenario_id: str | None) -> str:
    """Return the scikit-learn model class name for a scenario."""
    return MODEL_TECHNICAL_NAMES.get(scenario_id or "", MODEL_TECHNICAL_NAMES["regression"])
