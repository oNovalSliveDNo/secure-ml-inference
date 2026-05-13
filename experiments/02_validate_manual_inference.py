# experiments/02_validate_manual_inference.py
"""Experiment 02: Validate manual plaintext inference matches sklearn."""

import logging
import numpy as np
from app.data import load_dataset, split_dataset
from app.model import load_model, extract_linear_params, compute_manual_score, sigmoid
from app.inference import plaintext_inference, manual_plaintext_inference
from app.metrics import compare_predictions
from app.config import RANDOM_STATE, TEST_SIZE, THRESHOLD

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    logger.info("Validating manual inference...")
    # TODO: load model, test data, compare predictions
    logger.info("Validation complete.")


if __name__ == "__main__":
    main()
