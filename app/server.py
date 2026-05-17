# app/server.py
"""Backward-compatible server wrapper around encrypted linear scorer."""

from app.linear_scorer import EncryptedLinearScorer, Server

__all__ = ["EncryptedLinearScorer", "Server"]
