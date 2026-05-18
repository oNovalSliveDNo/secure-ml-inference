# api/main.py
"""FastAPI application entrypoint for model serving."""

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import phe as paillier
from fastapi import FastAPI, HTTPException

from app.config import SCALE
from app.crypto import deserialize_ciphertext, serialize_ciphertext
from app.encoding import encode_bias, encode_weights
from app.linear_scorer import EncryptedLinearScorer
from app.model import extract_linear_params, load_model
from app.schemas import EncryptedInferRequest, EncryptedInferResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "results/models/model.pkl"
REGRESSION_MODEL_PATH = "results/models/regression_model.pkl"
DEFAULT_SCENARIO_ID = "classification"
SUPPORTED_SCENARIO_IDS = ("classification", "regression")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model artifacts for all supported scenarios."""

    # --- Классификация ---
    pipeline_cls = load_model(MODEL_PATH)
    w_cls, b_cls = extract_linear_params(pipeline=pipeline_cls)
    w_int_cls = encode_weights(w=w_cls, scale=SCALE)
    b_int_cls = encode_bias(b=b_cls, scale=SCALE)

    class_names = ["malignant", "benign"]  # или из dataset

    scenarios: dict[str, dict[str, Any]] = {
        "classification": {
            "w_int": w_int_cls,
            "b_int": b_int_cls,
            "feature_count": int(w_int_cls.shape[0]),
        }
    }

    # --- Регрессия ---
    if Path(REGRESSION_MODEL_PATH).exists():
        pipeline_reg = load_model(REGRESSION_MODEL_PATH)
        # Достаём Ridge-регрессор напрямую (в обход extract_linear_params)
        ridge = pipeline_reg.named_steps["regressor"]
        w_reg = ridge.coef_.ravel()
        b_reg = float(ridge.intercept_)
        w_int_reg = encode_weights(w=w_reg, scale=SCALE)
        b_int_reg = encode_bias(b=b_reg, scale=SCALE)
        scenarios["regression"] = {
            "w_int": w_int_reg,
            "b_int": b_int_reg,
            "feature_count": int(w_int_reg.shape[0]),
        }
        logger.info("Regression model loaded with %d features", w_int_reg.shape[0])
    else:
        logger.warning("Regression model not found at %s", REGRESSION_MODEL_PATH)

    app.state.scenarios = scenarios
    app.state.feature_count = int(scenarios["classification"]["feature_count"])
    app.state.classes = class_names

    logger.info(
        "Server initialized with scenarios=%s, features=%s and classes=%s",
        list(app.state.scenarios.keys()),
        app.state.feature_count,
        class_names,
    )
    yield


app = FastAPI(title="Secure ML Inference API", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/model/info")
async def model_info() -> dict[str, object]:
    """Return model metadata (feature count, class labels)."""
    return {
        "feature_count": int(app.state.feature_count),
        "classes": list(app.state.classes),
    }


@app.post("/infer/encrypted", response_model=EncryptedInferResponse)
async def infer_encrypted(request: EncryptedInferRequest) -> EncryptedInferResponse:
    """Compute encrypted linear score from encrypted feature vector."""
    if not hasattr(app.state, "scenarios"):
        raise HTTPException(status_code=503, detail="Server is not initialized")

    scenario_id = request.scenario_id or DEFAULT_SCENARIO_ID
    if scenario_id not in app.state.scenarios:
        raise HTTPException(status_code=400, detail=f"Unknown scenario_id: {scenario_id}")
    scenario = app.state.scenarios[scenario_id]

    if request.feature_count != len(request.encrypted_features):
        raise HTTPException(status_code=400, detail="feature_count does not match payload")

    if request.feature_count != int(scenario["feature_count"]):
        raise HTTPException(status_code=400, detail="feature_count does not match model")

    if request.scale != SCALE:
        raise HTTPException(status_code=400, detail="scale does not match server configuration")

    try:
        public_key = paillier.PaillierPublicKey(n=int(request.public_key_n))
        encrypted_features = [
            deserialize_ciphertext(public_key=public_key, ciphertext_str=ciphertext)
            for ciphertext in request.encrypted_features
        ]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid encrypted payload: {exc}") from exc

    request_server = EncryptedLinearScorer(
        w_int=scenario["w_int"],
        b_int=scenario["b_int"],
        public_key=public_key,
    )

    start = time.perf_counter()
    encrypted_score = request_server.compute_encrypted_score(enc_x=encrypted_features)
    compute_ms = (time.perf_counter() - start) * 1000.0

    return EncryptedInferResponse(
        encrypted_score=serialize_ciphertext(encrypted_number=encrypted_score),
        server_compute_ms=compute_ms,
        feature_count=request.feature_count,
    )
