# experiments/01_train_baseline.py
"""Experiment 01: Train baseline model and save artifacts."""

import logging
from pathlib import Path
from app.data import load_dataset, split_dataset
from app.model import train_baseline_model, save_model, extract_linear_params
from app.config import RANDOM_STATE, TEST_SIZE
from app.metrics import classification_metrics
import joblib
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    logger.info("Starting baseline training...")
    # TODO: load data, split, train model, compute metrics, save model, scaler, weights, metadata
    logger.info("Baseline training completed.")


if __name__ == "__main__":
    main()
