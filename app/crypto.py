# app/crypto.py
"""Paillier encryption wrapper using the python-paillier library."""

import phe as paillier


def generate_keys(
    n_length: int = 1024,
) -> tuple[paillier.PaillierPublicKey, paillier.PaillierPrivateKey]:
    """
    Generate a Paillier keypair.

    Args:
        n_length: Bit length of the modulus n.

    Returns:
        (public_key, private_key) tuple.
    """
    # TODO: implement
    raise NotImplementedError


def encrypt_vector(
    public_key: paillier.PaillierPublicKey,
    x_int: list[int],
) -> list[paillier.EncryptedNumber]:
    """
    Encrypt a list of integers under the given public key.

    Args:
        public_key: Paillier public key.
        x_int: List of integer features.

    Returns:
        List of encrypted numbers.
    """
    # TODO: implement
    raise NotImplementedError


def decrypt_score(
    private_key: paillier.PaillierPrivateKey,
    encrypted_score: paillier.EncryptedNumber,
) -> int:
    """
    Decrypt an encrypted score to recover the integer.

    Args:
        private_key: Paillier private key.
        encrypted_score: Encrypted integer score.

    Returns:
        Decrypted integer score.
    """
    # TODO: implement
    raise NotImplementedError


def serialize_ciphertext(encrypted_number: paillier.EncryptedNumber) -> str:
    """Serialize an encrypted number to a string for JSON transfer."""
    # TODO: implement (e.g., str(encrypted_number.ciphertext()))
    raise NotImplementedError


def deserialize_ciphertext(
    public_key: paillier.PaillierPublicKey,
    ciphertext_str: str,
) -> paillier.EncryptedNumber:
    """Reconstruct an EncryptedNumber from string and public key."""
    # TODO: implement
    raise NotImplementedError


def estimate_ciphertext_size(encrypted_number: paillier.EncryptedNumber) -> int:
    """Return size of ciphertext in bytes."""
    # TODO: implement
    raise NotImplementedError
