# app/config.py
"""Central configuration for secure-ml-inference with environment overrides."""

import os

RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))
SCALE: int = int(os.getenv("SCALE", "10000"))
KEY_LENGTH: int = int(os.getenv("KEY_LENGTH", "1024"))
THRESHOLD: float = float(os.getenv("THRESHOLD", "0.5"))
FEATURE_COUNTS: list[int] = [
    int(value.strip())
    for value in os.getenv("FEATURE_COUNTS", "5,10,20,30").split(",")
    if value.strip()
]
N_BENCHMARK_RUNS: int = int(os.getenv("N_BENCHMARK_RUNS", "30"))
