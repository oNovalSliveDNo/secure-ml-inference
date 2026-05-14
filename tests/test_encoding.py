"""Unit tests for fixed-point encoding utilities."""

import numpy as np

from app.encoding import (
    decode_score,
    encode_bias,
    encode_vector,
    encode_weights,
    encoded_plaintext_score,
)


def test_encode_vector_and_weights_are_reversible_with_tolerance() -> None:
    scale = 10_000
    values = np.array([0.1234, -1.5678, 2.0, -0.00009], dtype=np.float64)

    encoded = encode_vector(values, scale)
    restored = encoded.astype(np.float64) / scale

    assert encoded.dtype == np.int64
    assert np.allclose(restored, values, atol=0.5 / scale)


def test_decode_score_matches_manual_double_scale_rule() -> None:
    scale = 10_000
    x = np.array([0.2, -1.1, 3.4], dtype=np.float64)
    w = np.array([1.5, -0.2, 0.75], dtype=np.float64)
    b = -0.31

    x_int = encode_vector(x, scale)
    w_int = encode_weights(w, scale)
    b_int = encode_bias(b, scale)
    score_int = int(np.dot(x_int, w_int) + b_int)

    assert decode_score(score_int, scale) == score_int / (scale * scale)


def test_encoded_plaintext_score_close_to_float_score() -> None:
    scale = 10_000
    x = np.array([1.25, -0.7, 0.55], dtype=np.float64)
    w = np.array([0.45, 1.2, -0.1], dtype=np.float64)
    b = 0.03

    float_score = float(np.dot(x, w) + b)
    encoded_score = encoded_plaintext_score(x=x, w=w, b=b, scale=scale)

    assert abs(float_score - encoded_score) < 1e-3
