# tests/test_paillier_reference.py

from app.paillier_reference import (
    PaillierKeypair,
    homomorphic_add,
    homomorphic_scalar_mul,
)


def test_roundtrip_for_multiple_messages() -> None:
    keypair = PaillierKeypair.generate()

    for m in [0, 1, 2, 10, keypair.n - 1]:
        c = keypair.encrypt(m)
        assert keypair.decrypt(c) == m


def test_homomorphic_addition() -> None:
    keypair = PaillierKeypair.generate()

    a, b = 42, 123
    c_a = keypair.encrypt(a)
    c_b = keypair.encrypt(b)

    c_add = homomorphic_add(c_a, c_b, keypair)
    assert keypair.decrypt(c_add) == (a + b) % keypair.n


def test_homomorphic_scalar_multiplication() -> None:
    keypair = PaillierKeypair.generate()

    a, k = 37, 9
    c_a = keypair.encrypt(a)

    c_mul = homomorphic_scalar_mul(c_a, k, keypair)
    assert keypair.decrypt(c_mul) == (a * k) % keypair.n


def test_encrypt_rejects_message_out_of_range() -> None:
    keypair = PaillierKeypair.generate()

    try:
        keypair.encrypt(keypair.n)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for m >= n")


def test_encrypt_rejects_invalid_r() -> None:
    keypair = PaillierKeypair.generate()

    # r = n заведомо вне диапазона 1 <= r < n
    try:
        keypair.encrypt(5, r=keypair.n)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid r")
