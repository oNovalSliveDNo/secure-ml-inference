# app/client.py
"""Client-side logic for secure inference."""

from typing import Any

import numpy as np
import phe as paillier


class Client:
    """
    Represents the data owner.

    Responsibilities:
    - Apply StandardScaler (provided externally)
    - Encode features
    - Generate keypair
    - Encrypt features
    - Decrypt result and compute final prediction
    """

    def __init__(self, scaler: Any, scale: int, key_length: int = 1024) -> None:
        """
        Initialize client with a pre-fitted StandardScaler.

        Args:
            scaler: Fitted StandardScaler instance.
            scale: Fixed-point scaling factor.
            key_length: Paillier key length in bits.
        """
        # TODO: store scaler, scale, key_length; generate keys
        raise NotImplementedError

    def preprocess(self, raw_x: np.ndarray) -> np.ndarray:
        """Scale raw features using the stored scaler."""
        # TODO: implement
        raise NotImplementedError

    def encode(self, x_scaled: np.ndarray) -> np.ndarray:
        """Encode scaled float features to integers."""
        # TODO: use encode_vector
        raise NotImplementedError

    def encrypt(self, x_int: np.ndarray) -> list[paillier.EncryptedNumber]:
        """Encrypt integer feature vector with public key."""
        # TODO: use encrypt_vector
        raise NotImplementedError

    def decrypt_and_predict(self, encrypted_score: paillier.EncryptedNumber) -> tuple[int, float]:
        """
        Decrypt score, apply sigmoid and threshold.

        Returns:
            (prediction, probability) where prediction is 0 or 1.
        """
        # TODO: implement
        raise NotImplementedError
