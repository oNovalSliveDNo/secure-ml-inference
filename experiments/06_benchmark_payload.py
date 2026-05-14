# experiments/06_benchmark_payload.py
"""Experiment 06: Measure payload sizes of plaintext vs encrypted requests/responses."""

from __future__ import annotations

import csv
import logging
import warnings
from pathlib import Path

from app.client import Client
from app.config import KEY_LENGTH, SCALE
from app.crypto import serialize_ciphertext
from app.data import load_dataset
from app.encoding import encode_bias, encode_weights
from app.metrics import measure_payload_size
from app.model import extract_linear_params, load_model
from app.server import Server

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


MODEL_PATH = Path("results/models/model.pkl")
TABLES_DIR = Path("results/tables")
PAYLOAD_CSV_PATH = TABLES_DIR / "payload_metrics.csv"


def main() -> None:
    """Benchmark JSON payload growth from plaintext to encrypted protocol."""
    logger.info("Benchmarking payload sizes...")

    model = load_model(str(MODEL_PATH))
    features, _ = load_dataset()
    sample_raw = features.iloc[0].to_numpy(dtype=float)

    scaler = model.named_steps["scaler"]
    w, b = extract_linear_params(model)

    client = Client(scaler=scaler, scale=SCALE, key_length=KEY_LENGTH)
    server = Server(
        w_int=encode_weights(w=w, scale=SCALE),
        b_int=encode_bias(b=b, scale=SCALE),
        public_key=client.public_key,
    )

    # Request payloads
    plaintext_request = sample_raw.tolist()  # JSON-list of float values.

    x_scaled = client.preprocess(sample_raw.reshape(1, -1))
    x_int = client.encode(x_scaled)
    enc_x = client.encrypt(x_int)
    encrypted_request = {
        "public_key_n": str(client.public_key.n),
        "encrypted_features": [serialize_ciphertext(value) for value in enc_x],
        "scale": SCALE,
    }

    plaintext_request_bytes = measure_payload_size(plaintext_request)
    encrypted_request_bytes = measure_payload_size(encrypted_request)

    # Response payloads
    plaintext_prob = float(model.predict_proba(sample_raw.reshape(1, -1))[:, 1][0])
    plaintext_response = {"score": plaintext_prob}

    encrypted_score = server.compute_encrypted_score(enc_x)
    encrypted_response = {"encrypted_score": serialize_ciphertext(encrypted_score)}

    plaintext_response_bytes = measure_payload_size(plaintext_response)
    encrypted_response_bytes = measure_payload_size(encrypted_response)

    request_growth_factor = float(encrypted_request_bytes / plaintext_request_bytes)
    response_growth_factor = float(encrypted_response_bytes / plaintext_response_bytes)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with PAYLOAD_CSV_PATH.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "plaintext_request_bytes",
                "encrypted_request_bytes",
                "request_growth_factor",
                "plaintext_response_bytes",
                "encrypted_response_bytes",
                "response_growth_factor",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "plaintext_request_bytes": plaintext_request_bytes,
                "encrypted_request_bytes": encrypted_request_bytes,
                "request_growth_factor": request_growth_factor,
                "plaintext_response_bytes": plaintext_response_bytes,
                "encrypted_response_bytes": encrypted_response_bytes,
                "response_growth_factor": response_growth_factor,
            }
        )

    logger.info("Request growth factor: %.4f", request_growth_factor)
    logger.info("Response growth factor: %.4f", response_growth_factor)
    logger.info("Saved payload metrics to %s", PAYLOAD_CSV_PATH)
    logger.info("Payload benchmark completed.")


if __name__ == "__main__":
    main()
