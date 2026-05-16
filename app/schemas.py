# app/schemas.py
"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field


class EncryptedInferRequest(BaseModel):
    """Request body for /infer/encrypted."""

    public_key_n: str
    encrypted_features: list[str] = Field(min_length=1)
    scale: int = Field(gt=0)
    feature_count: int = Field(gt=0)


class EncryptedInferResponse(BaseModel):
    """Response body for /infer/encrypted."""

    encrypted_score: str
    server_compute_ms: float
    feature_count: int
