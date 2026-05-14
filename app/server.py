# app/server.py
"""Server-side logic for secure inference."""

import numpy as np
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
    ) -> None:
        """
        Initialize server with encoded model parameters and a public key.

        Args:
            w_int: Encoded weight vector.
            b_int: Encoded bias.
            public_key: Client's public key for encrypting bias.
        """
        self.w_int: np.ndarray = np.asarray(w_int, dtype=np.int64)
        self.b_int: int = int(b_int)
        self.public_key: paillier.PaillierPublicKey = public_key
        self.enc_b: paillier.EncryptedNumber = self.public_key.encrypt(self.b_int)

    def compute_encrypted_score(
        self,
        enc_x: list[paillier.EncryptedNumber],
    ) -> paillier.EncryptedNumber:
        """
        Compute Enc(z) = sum(enc_x_i * w_i) + Enc(b_int).

        Args:
            enc_x: List of encrypted feature values.

        Returns:
            Encrypted linear score.

        Raises:
            ValueError: If feature length does not match model weight length.
        """
        if len(enc_x) != len(self.w_int):
            raise ValueError("enc_x length must match w_int length")

        enc_score: paillier.EncryptedNumber = self.enc_b
        for enc_feature, weight in zip(enc_x, self.w_int, strict=True):
            enc_score = enc_score + (enc_feature * int(weight))
        return enc_score
