"""Protocol trace privacy tests for encrypted inference UI payloads."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from app.config import SCALE
from ui.protocol_view import build_server_panel_data

ALLOWED_REQUEST_FIELDS = {
    "public_key_n",
    "encrypted_features",
    "scale",
    "scenario_id",
    "feature_count",
}

FORBIDDEN_TRACE_FIELDS = {
    "x_raw",
    "raw",
    "x_scaled",
    "scaled",
    "x_int",
    "private_key",
    "secret_key",
    "true_label",
    "y_true",
    "baseline",
    "baseline_prediction",
}


def _request_payload() -> dict[str, Any]:
    """Build the request trace shape used for POST /infer/encrypted."""
    return {
        "public_key_n": "123456789",
        "encrypted_features": ["111", "222", "333"],
        "scale": SCALE,
        "scenario_id": "classification",
        "feature_count": 3,
    }


def test_encrypted_infer_request_trace_contains_only_allowed_fields() -> None:
    payload = _request_payload()

    assert set(payload) == ALLOWED_REQUEST_FIELDS
    assert FORBIDDEN_TRACE_FIELDS.isdisjoint(payload)


def test_server_panel_data_excludes_plaintext_features_and_private_key() -> None:
    payload = _request_payload()
    result = {
        "request_payload": payload,
        "x_raw": [10.0, 20.0, 30.0],
        "raw": [10.0, 20.0, 30.0],
        "x_scaled": [0.1, 0.2, 0.3],
        "scaled": [0.1, 0.2, 0.3],
        "x_int": [1000, 2000, 3000],
        "private_key": "must-stay-client-side",
        "secret_key": "must-stay-client-side",
        "true_label": 1,
        "y_true": 1,
        "baseline": 0.91,
        "baseline_prediction": 1,
        "encrypted_score": "999",
        "server_compute_ms": 1.25,
    }
    scenario = {"w": [1.0, 2.0, 3.0]}
    sample = pd.Series([10.0, 20.0, 30.0], index=["a", "b", "c"])

    server_data = build_server_panel_data(
        result=result,
        scenario=scenario,
        sample=sample,
        scale=SCALE,
    )

    assert set(server_data["request_payload"]) == ALLOWED_REQUEST_FIELDS
    assert FORBIDDEN_TRACE_FIELDS.isdisjoint(server_data)
    assert FORBIDDEN_TRACE_FIELDS.isdisjoint(server_data["request_payload"])
    assert "features" not in server_data
    assert "sample" not in server_data
    assert "private_key" not in server_data


def test_control_encoded_score_matches_decrypted_phe_score() -> None:
    """The unencrypted fixed-point control path must match the decrypted PHE score."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    from app.client import Client
    from app.crypto import decrypt_score
    from app.encoding import decode_score, encode_bias, encode_weights, encoded_plaintext_score
    from app.linear_scorer import Server

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
    sample = np.array([1.2, 0.3, 1.9], dtype=np.float64)
    weights = np.array([0.8, -1.2, 0.5], dtype=np.float64)
    bias = -0.15

    client = Client(scaler=scaler, scale=scale, key_length=512)
    server = Server(
        w_int=encode_weights(w=weights, scale=scale),
        b_int=encode_bias(b=bias, scale=scale),
        public_key=client.public_key,
    )

    x_scaled = client.preprocess(sample.reshape(1, -1)).reshape(-1)
    x_int = client.encode(x_scaled)
    encrypted_score = server.compute_encrypted_score(client.encrypt(x_int))
    decrypted_phe_score = decode_score(
        score_int=decrypt_score(client.private_key, encrypted_score),
        scale=scale,
    )
    control_encoded_score = encoded_plaintext_score(
        x=x_scaled,
        w=weights,
        b=bias,
        scale=scale,
    )

    assert decrypted_phe_score == pytest.approx(control_encoded_score, abs=1e-12)
