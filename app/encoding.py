# app/encoding.py
"""Fixed-point encoding and decoding for secure integer arithmetic."""

import numpy as np


def encode_vector(x: np.ndarray, scale: int) -> np.ndarray:
    """
    Convert float vector to integer representation: x_int = round(x * scale).

    Args:
        x: Input float vector.
        scale: Scaling factor.

    Returns:
        Integer vector (int64).
    """
    x_array: np.ndarray = np.asarray(x, dtype=np.float64)
    encoded: np.ndarray = np.rint(x_array * scale).astype(np.int64)
    return encoded


def encode_weights(w: np.ndarray, scale: int) -> np.ndarray:
    """
    Encode weight vector similarly: w_int = round(w * scale).

    Args:
        w: Weight coefficients.
        scale: Scaling factor.

    Returns:
        Integer weight vector.
    """
    w_array: np.ndarray = np.asarray(w, dtype=np.float64)
    encoded: np.ndarray = np.rint(w_array * scale).astype(np.int64)
    return encoded


def encode_bias(b: float, scale: int) -> int:
    """
    Encode bias with double scaling: b_int = round(b * scale * scale).

    Args:
        b: Bias term.
        scale: Scaling factor.

    Returns:
        Encoded bias integer.
    """
    return int(np.rint(float(b) * scale * scale))


def decode_score(score_int: int, scale: int) -> float:
    """
    Decode integer score back to float: z = score_int / (scale * scale).

    Args:
        score_int: Encrypted linear score after decryption.
        scale: Scaling factor.

    Returns:
        Real-valued linear score.
    """
    return float(score_int) / float(scale * scale)


def encoded_plaintext_score(
    x: np.ndarray,
    w: np.ndarray,
    b: float,
    scale: int,
) -> float:
    """
    Compute linear score using fixed-point encoding but without encryption.

    Args:
        x: Scaled feature vector (float).
        w: Weight vector.
        b: Bias.
        scale: Scaling factor.

    Returns:
        Real-valued score (after decoding).
    """
    x_int: np.ndarray = encode_vector(x=x, scale=scale)
    w_int: np.ndarray = encode_weights(w=w, scale=scale)
    b_int: int = encode_bias(b=b, scale=scale)

    score_int: int = int(np.dot(x_int, w_int) + b_int)
    return decode_score(score_int=score_int, scale=scale)
