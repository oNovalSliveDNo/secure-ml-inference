# experiments/03_run_encoded_inference.py
"""Experiment 03: Evaluate encoded plaintext inference quality."""

import logging
import numpy as np
from app.data import load_dataset, split_dataset
from app.model import load_model, extract_linear_params
from app.inference import plaintext_inference, encoded_plaintext_inference
from app.metrics import classification_metrics, compare_predictions
from app.config import RANDOM_STATE, TEST_SIZE, SCALE, THRESHOLD

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    logger.info("Running encoded plaintext inference...")
    # TODO: load model, test data, run encoded inference, compute metrics, save to CSV
    logger.info("Encoded plaintext inference completed.")


if __name__ == "__main__":
    main()
