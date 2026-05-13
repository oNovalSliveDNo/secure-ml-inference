# app/config.py
"""
Central configuration for the secure-ml-inference project.
Uses constants and environment variables with sensible defaults.
"""

import os

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
SCALE: int = int(os.getenv("SCALE", 10000))
KEY_LENGTH: int = int(os.getenv("KEY_LENGTH", 1024))
THRESHOLD: float = 0.5
FEATURE_COUNTS: list[int] = [5, 10, 20, 30]
N_BENCHMARK_RUNS: int = int(os.getenv("N_BENCHMARK_RUNS", 30))
