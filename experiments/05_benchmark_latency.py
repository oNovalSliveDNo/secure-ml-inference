# experiments/05_benchmark_latency.py
"""Experiment 05: Benchmark latency components of PHE inference."""

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    logger.info("Benchmarking latency...")
    # TODO: measure times: preprocessing, encoding, keygen, encryption, server, decryption, total
    # Save to results/tables/latency_metrics.csv
    logger.info("Latency benchmark completed.")


if __name__ == "__main__":
    main()
