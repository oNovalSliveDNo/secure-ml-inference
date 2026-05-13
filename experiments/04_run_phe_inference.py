# experiments/04_run_phe_inference.py
"""Experiment 04: Run PHE inference on test set and measure quality."""

import logging
import numpy as np
from app.data import load_dataset, split_dataset
from app.model import load_model, extract_linear_params
from app.client import Client
from app.server import Server
from app.inference import phe_inference_batch
from app.metrics import classification_metrics, compare_predictions
from app.config import RANDOM_STATE, TEST_SIZE, SCALE, KEY_LENGTH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    logger.info("Starting PHE inference...")
    # TODO: load model, prepare client and server, run inference on test set
    logger.info("PHE inference completed.")


if __name__ == "__main__":
    main()
