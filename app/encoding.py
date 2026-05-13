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
    # TODO: implement
    raise NotImplementedError


def encode_weights(w: np.ndarray, scale: int) -> np.ndarray:
    """
    Encode weight vector similarly: w_int = round(w * scale).

    Args:
        w: Weight coefficients.
        scale: Scaling factor.

    Returns:
        Integer weight vector.
    """
    # TODO: implement
    raise NotImplementedError


def encode_bias(b: float, scale: int) -> int:
    """
    Encode bias with double scaling: b_int = round(b * scale * scale).

    Args:
        b: Bias term.
        scale: Scaling factor.

    Returns:
        Encoded bias integer.
    """
    # TODO: implement
    raise NotImplementedError


def decode_score(score_int: int, scale: int) -> float:
    """
    Decode integer score back to float: z = score_int / (scale * scale).

    Args:
        score_int: Encrypted linear score after decryption.
        scale: Scaling factor.

    Returns:
        Real-valued linear score.
    """
    # TODO: implement
    raise NotImplementedError


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
    # TODO: implement using encode/decode functions
    raise NotImplementedError
