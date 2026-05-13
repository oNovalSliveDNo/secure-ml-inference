# experiments/06_benchmark_payload.py
"""Experiment 06: Measure payload sizes of plaintext vs encrypted requests/responses."""

import logging
import json
import sys
import pandas as pd
from pathlib import Path
from app.crypto import encrypt_vector, serialize_ciphertext
from app.config import SCALE, KEY_LENGTH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    logger.info("Benchmarking payload sizes...")
    # TODO: generate sample data, measure plaintext/encrypted sizes, save to CSV
    logger.info("Payload benchmark completed.")


if __name__ == "__main__":
    main()
