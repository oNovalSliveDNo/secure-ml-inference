# experiments/01_train_baseline.py
"""Experiment 01: Train baseline model and save artifacts."""

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    logger.info("Starting baseline training...")
    # TODO: load data, split, train model, compute metrics, save model, scaler, weights, metadata
    logger.info("Baseline training completed.")


if __name__ == "__main__":
    main()
