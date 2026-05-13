# app/schemas.py
"""Pydantic models for API request/response validation."""

from pydantic import BaseModel


class EncryptedInferRequest(BaseModel):
    """Request body for /infer/encrypted."""

    public_key_n: str
    encrypted_features: list[str]
    scale: int
    feature_count: int


class EncryptedInferResponse(BaseModel):
    """Response body for /infer/encrypted."""

    encrypted_score: str
    server_compute_ms: float
    feature_count: int
