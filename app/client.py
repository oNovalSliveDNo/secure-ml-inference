# app/client.py
"""Client-side logic for secure inference."""

from typing import Any

import numpy as np
import pandas as pd
import phe as paillier

from app.config import THRESHOLD
from app.crypto import decrypt_score, encrypt_vector, generate_keys
from app.encoding import decode_score, encode_vector


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
        self.scaler: Any = scaler
        self.scale: int = int(scale)
        self.key_length: int = int(key_length)
        self.public_key, self.private_key = generate_keys(n_length=self.key_length)

    def preprocess(self, raw_x: Any) -> np.ndarray:
        """
        Scale one raw sample using the stored scaler.

        Accepts a DataFrame (one row) or a numpy array.
        If a numpy array is given, it is automatically wrapped into a DataFrame
        using the feature names the scaler was fitted on, to suppress warnings.

        Args:
            raw_x: One sample (DataFrame or array with shape (1, n_features)).

        Returns:
            One-dimensional scaled feature vector.
        """
        if isinstance(raw_x, np.ndarray):
            # Восстанавливаем имена признаков, чтобы scaler не ругался
            if hasattr(self.scaler, "feature_names_in_"):
                raw_x = pd.DataFrame(raw_x, columns=self.scaler.feature_names_in_)
            else:
                raw_x = pd.DataFrame(raw_x)
        x_scaled = self.scaler.transform(raw_x)
        return np.asarray(x_scaled[0], dtype=np.float64)

    def encode(self, x_scaled: np.ndarray) -> np.ndarray:
        """Encode scaled float features to integers."""
        return encode_vector(x=x_scaled, scale=self.scale)

    def encrypt(self, x_int: np.ndarray) -> list[paillier.EncryptedNumber]:
        """
        Encrypt integer feature vector with the stored public key.

        Args:
            x_int: Encoded feature vector.

        Returns:
            Encrypted feature list.
        """
        x_list: list[int] = [int(v) for v in np.asarray(x_int, dtype=np.int64).tolist()]
        return encrypt_vector(public_key=self.public_key, x_int=x_list)

    def decrypt_and_predict(self, encrypted_score: paillier.EncryptedNumber) -> tuple[int, float]:
        """
        Decrypt score, apply sigmoid and threshold.

        Args:
            encrypted_score: Encrypted linear score from server.

        Returns:
            (prediction, probability) where prediction is 0 or 1.
        """
        score_int: int = decrypt_score(self.private_key, encrypted_score)
        z: float = decode_score(score_int=score_int, scale=self.scale)
        probability: float = float(1.0 / (1.0 + np.exp(-z)))
        prediction: int = int(probability >= THRESHOLD)
        return prediction, probability
