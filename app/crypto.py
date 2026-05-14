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
    public_key, private_key = paillier.generate_paillier_keypair(n_length=n_length)
    return public_key, private_key


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

    Raises:
        ValueError: If ``x_int`` is not a list of integers.
    """
    if not isinstance(x_int, list) or not all(isinstance(v, int) for v in x_int):
        raise ValueError("x_int must be a list[int]")

    return [public_key.encrypt(v) for v in x_int]


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
    value = private_key.decrypt(encrypted_score)
    return int(value)


def serialize_ciphertext(encrypted_number: paillier.EncryptedNumber) -> str:
    """
    Serialize an encrypted number to a string for JSON transfer.

    Args:
        encrypted_number: Encrypted number to serialize.

    Returns:
        Decimal string representation of the ciphertext integer.
    """
    return str(encrypted_number.ciphertext())


def deserialize_ciphertext(
    public_key: paillier.PaillierPublicKey,
    ciphertext_str: str,
) -> paillier.EncryptedNumber:
    """
    Reconstruct an EncryptedNumber from string and public key.

    Args:
        public_key: Paillier public key matching the ciphertext.
        ciphertext_str: Serialized decimal ciphertext.

    Returns:
        Reconstructed encrypted number.

    Raises:
        ValueError: If ``ciphertext_str`` is not a valid integer string.
    """
    ciphertext_int = int(ciphertext_str)
    return paillier.EncryptedNumber(public_key, ciphertext_int)


def estimate_ciphertext_size(encrypted_number: paillier.EncryptedNumber) -> int:
    """
    Estimate serialized ciphertext size in bytes.

    Args:
        encrypted_number: Encrypted number to estimate.

    Returns:
        Size in bytes of UTF-8 encoded serialized ciphertext.
    """
    serialized = serialize_ciphertext(encrypted_number)
    return len(serialized.encode("utf-8"))
