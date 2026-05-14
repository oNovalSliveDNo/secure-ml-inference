# tests/test_crypto.py
"""Unit tests for Paillier crypto helpers."""

from app.crypto import (
    decrypt_score,
    deserialize_ciphertext,
    encrypt_vector,
    generate_keys,
    serialize_ciphertext,
)


def test_encrypt_decrypt_single_value_roundtrip() -> None:
    public_key, private_key = generate_keys(n_length=512)

    value = 42
    encrypted_value = encrypt_vector(public_key, [value])[0]
    decrypted_value = decrypt_score(private_key, encrypted_value)

    assert decrypted_value == value


def test_serialize_deserialize_keeps_decryptability() -> None:
    public_key, private_key = generate_keys(n_length=512)

    value = -7
    encrypted_value = public_key.encrypt(value)
    serialized = serialize_ciphertext(encrypted_value)
    restored = deserialize_ciphertext(public_key, serialized)

    assert decrypt_score(private_key, restored) == value


def test_paillier_homomorphic_addition_and_scalar_multiplication() -> None:
    public_key, private_key = generate_keys(n_length=512)

    a = 12
    b = -5
    k = 3

    enc_a = public_key.encrypt(a)
    enc_b = public_key.encrypt(b)

    assert decrypt_score(private_key, enc_a + enc_b) == a + b
    assert decrypt_score(private_key, enc_a * k) == a * k
