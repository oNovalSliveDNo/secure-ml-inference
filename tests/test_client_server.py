"""Integration-style unit tests for client/server secure inference flow."""

import numpy as np
from sklearn.preprocessing import StandardScaler

from app.client import Client
from app.encoding import encode_bias, encode_weights
from app.inference import phe_inference_one
from app.server import Server


def test_client_server_single_sample_matches_manual_encoded_path() -> None:
    scale = 10_000

    x_train = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 1.0, 0.0],
            [1.5, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    scaler = StandardScaler().fit(x_train)

    client = Client(scaler=scaler, scale=scale, key_length=512)
    w = np.array([0.8, -1.2, 0.5], dtype=np.float64)
    b = -0.15

    server = Server(
        w_int=encode_weights(w=w, scale=scale),
        b_int=encode_bias(b=b, scale=scale),
        public_key=client.public_key,
    )

    sample = np.array([1.2, 0.3, 1.9], dtype=np.float64)
    pred, prob = phe_inference_one(client=client, server=server, x_raw=sample)

    x_scaled = scaler.transform(sample.reshape(1, -1))[0]
    z_expected = float(
        (np.dot(np.rint(x_scaled * scale), np.rint(w * scale)) + np.rint(b * scale * scale))
        / (scale * scale)
    )
    prob_expected = float(1.0 / (1.0 + np.exp(-z_expected)))
    pred_expected = int(prob_expected >= 0.5)

    assert pred == pred_expected
    assert abs(prob - prob_expected) < 1e-8
