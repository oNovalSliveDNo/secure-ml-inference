# experiments/07_benchmark_feature_scaling.py
"""Experiment 07: Benchmark impact of feature count on time and size."""

import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from app.client import Client
from app.server import Server
from app.model import load_model, extract_linear_params
from app.config import FEATURE_COUNTS, SCALE, KEY_LENGTH, N_BENCHMARK_RUNS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    logger.info("Feature scaling benchmark...")
    # TODO: for each k in FEATURE_COUNTS, measure time & size, save CSV and plots
    logger.info("Feature scaling benchmark completed.")


if __name__ == "__main__":
    main()
