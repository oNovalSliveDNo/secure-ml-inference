# experiments/02_validate_manual_inference.py
"""Experiment 02: Validate manual plaintext inference matches sklearn."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.config import RANDOM_STATE, TEST_SIZE
from app.data import load_dataset, split_dataset
from app.inference import manual_plaintext_inference, plaintext_inference
from app.model import compute_manual_score, extract_linear_params, load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_PATH = Path("results/models/model.pkl")
SCORE_TOLERANCE = 1e-9


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main() -> None:
    """
    Validate that manual plaintext inference exactly matches sklearn baseline.
    """

    logger.info("Validating manual inference...")

    model = load_model(str(MODEL_PATH))
    features, target = load_dataset()
    _, x_test, _, _ = split_dataset(
        features=features,
        target=target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    baseline_pred, _ = plaintext_inference(model=model, x_test=x_test)

    scaler = model.named_steps["scaler"]
    x_scaled = scaler.transform(x_test)
    w, b = extract_linear_params(model)

    manual_pred, _ = manual_plaintext_inference(x_scaled=x_scaled, w=w, b=b)

    baseline_scores = np.asarray(
        model.named_steps["logreg"].decision_function(x_scaled), dtype=np.float64
    )
    manual_scores = compute_manual_score(x=x_scaled, w=w, b=b)

    match_rate = float(np.mean(baseline_pred == manual_pred))
    max_score_diff = float(np.max(np.abs(baseline_scores - manual_scores)))

    logger.info("match_rate=%.10f", match_rate)
    logger.info("max_score_diff=%.12e", max_score_diff)

    if match_rate == 1.0 and max_score_diff < SCORE_TOLERANCE:
        logger.info("Manual plaintext inference validation passed.")

    logger.info("Validation complete.")


if __name__ == "__main__":
    main()
