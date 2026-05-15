# app/paillier_reference.py
"""Учебная реализация Paillier, не для production/реальной криптозащиты.

Модуль демонстрирует базовые формулы Paillier:
- шифрование: c = g^m * r^n mod n^2
- расшифрование: m = L(c^λ mod n^2) * μ mod n
- гомоморфное сложение и скалярное умножение в пространстве шифртекстов.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from math import gcd, lcm


def _l_function(u: int, n: int) -> int:
    """Функция L(u) = (u - 1) / n для схемы Paillier."""
    return (u - 1) // n


@dataclass(slots=True)
class PaillierKeypair:
    """Учебная реализация, не для production/реальной криптозащиты.

    Поля:
    - n, g: публичный ключ
    - lambda_, mu: приватные параметры для расшифрования
    - n_sq: предвычисленный модуль n^2 для операций над шифртекстами
    """

    n: int
    g: int
    lambda_: int
    mu: int
    n_sq: int

    @classmethod
    def generate(
        cls,
        p: int | None = None,
        q: int | None = None,
        *,
        use_crypto_util: bool = False,
        bits: int = 128,
    ) -> PaillierKeypair:
        """Генерирует учебную пару ключей.

        По умолчанию используются фиксированные маленькие простые p=17, q=19
        для детерминированной демонстрации. При use_crypto_util=True выполняется
        попытка импортировать Crypto.Util.number и сгенерировать случайные простые.
        """
        if p is None:
            p = 17
        if q is None:
            q = 19

        if use_crypto_util:
            try:
                from Crypto.Util.number import getPrime
            except ImportError as exc:
                raise ImportError(
                    "Crypto.Util.number недоступен; установите pycryptodome "
                    "или используйте демонстрационные p/q."
                ) from exc

            p = getPrime(bits)
            q = getPrime(bits)
            while q == p:
                q = getPrime(bits)

        n = p * q
        n_sq = n * n
        lambda_ = lcm(p - 1, q - 1)

        # В учебной схеме выбираем g = n + 1.
        g = n + 1

        # μ = (L(g^λ mod n^2))^{-1} mod n
        g_lambda_mod = pow(g, lambda_, n_sq)
        l_value = _l_function(g_lambda_mod, n)
        mu = pow(l_value, -1, n)

        return cls(n=n, g=g, lambda_=lambda_, mu=mu, n_sq=n_sq)

    def encrypt(self, m: int, r: int | None = None) -> int:
        """Шифрует m по формуле c = g^m * r^n mod n^2."""
        if not 0 <= m < self.n:
            raise ValueError(f"Message out of range: expected 0 <= m < n ({self.n}).")

        # r ∈ Z_n* (взаимно просто с n)
        if r is None:
            while True:
                candidate = secrets.randbelow(self.n - 1) + 1
                if gcd(candidate, self.n) == 1:
                    r = candidate
                    break
        elif not (1 <= r < self.n and gcd(r, self.n) == 1):
            raise ValueError("r must satisfy 1 <= r < n and gcd(r, n) = 1.")

        # c = g^m * r^n mod n^2
        gm = pow(self.g, m, self.n_sq)
        rn = pow(r, self.n, self.n_sq)
        return (gm * rn) % self.n_sq

    def decrypt(self, c: int) -> int:
        """Расшифрует c по формуле m = L(c^λ mod n^2) * μ mod n."""
        if not 0 <= c < self.n_sq:
            raise ValueError(f"Ciphertext out of range: expected 0 <= c < n^2 ({self.n_sq}).")

        # u = c^λ mod n^2
        u = pow(c, self.lambda_, self.n_sq)
        # L(u) = (u - 1) / n
        l_value = _l_function(u, self.n)
        # m = L(u) * μ mod n
        return (l_value * self.mu) % self.n


def _extract_public_key(public_key: tuple[int, int] | PaillierKeypair) -> tuple[int, int, int]:
    """Унифицирует извлечение (n, g, n_sq) из tuple или PaillierKeypair."""
    if isinstance(public_key, PaillierKeypair):
        return public_key.n, public_key.g, public_key.n_sq

    n, g = public_key
    return n, g, n * n


def homomorphic_add(c1: int, c2: int, public_key: tuple[int, int] | PaillierKeypair) -> int:
    """Гомоморфное сложение: c_add = c1 * c2 mod n^2."""
    _n, _g, n_sq = _extract_public_key(public_key)  # <-- было n, стало _n

    if not (0 <= c1 < n_sq and 0 <= c2 < n_sq):
        raise ValueError("Ciphertexts must be in range 0 <= c < n^2.")

    return (c1 * c2) % n_sq


def homomorphic_scalar_mul(c: int, k: int, public_key: tuple[int, int] | PaillierKeypair) -> int:
    """Скалярное умножение: c_mul = c^k mod n^2."""
    _n, _g, n_sq = _extract_public_key(public_key)  # <-- было n, стало _n

    if not 0 <= c < n_sq:
        raise ValueError("Ciphertext must be in range 0 <= c < n^2.")
    if k < 0:
        raise ValueError("k must be non-negative in this reference implementation.")

    return pow(c, k, n_sq)
