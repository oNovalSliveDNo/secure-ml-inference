# app/server.py
"""Server-side logic for secure inference."""

import numpy as np
from typing import List
import phe as paillier


class Server:
    """
    Represents the model owner.

    Responsibilities:
    - Hold encoded model weights
    - Accept encrypted feature vector
    - Compute encrypted linear score
    - Return encrypted score
    - Does NOT see plaintext features or private key
    """

    def __init__(
        self,
        w_int: np.ndarray,
        b_int: int,
        public_key: paillier.PaillierPublicKey,
    ):
        """
        Args:
            w_int: Encoded weight vector.
            b_int: Encoded bias.
            public_key: Client's public key for encrypting bias.
        """
        # TODO: store weights, bias, public key; pre-encrypt bias
        raise NotImplementedError

    def compute_encrypted_score(
        self,
        enc_x: List[paillier.EncryptedNumber],
    ) -> paillier.EncryptedNumber:
        """
        Compute Enc(z) = sum(enc_x_i * w_i) + Enc(b_int).

        Args:
            enc_x: List of encrypted feature values.

        Returns:
            Encrypted linear score.
        """
        # TODO: implement linear combination under encryption
        raise NotImplementedError
