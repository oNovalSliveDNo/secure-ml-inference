# api/main.py
"""FastAPI application entrypoint for model serving."""

import logging
import time

import phe as paillier
from fastapi import FastAPI, HTTPException
from sklearn.datasets import load_breast_cancer

from app.config import SCALE
from app.crypto import deserialize_ciphertext, serialize_ciphertext
from app.encoding import encode_bias, encode_weights
from app.model import extract_linear_params, load_model
from app.schemas import EncryptedInferRequest, EncryptedInferResponse
from app.server import Server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "results/models/model.pkl"

app = FastAPI(title="Secure ML Inference API")

# Server-side encoded model parameters are immutable after startup.


@app.on_event("startup")
async def startup_event() -> None:
    """Load model artifacts and initialize encrypted inference server state."""
    pipeline = load_model(MODEL_PATH)
    weights, bias = extract_linear_params(pipeline=pipeline)

    w_int = encode_weights(w=weights, scale=SCALE)
    b_int = encode_bias(b=bias, scale=SCALE)

    dataset = load_breast_cancer()
    class_names = [str(label) for label in dataset.target_names]

    bootstrap_key, _ = paillier.generate_paillier_keypair(n_length=1024)
    server = Server(w_int=w_int, b_int=b_int, public_key=bootstrap_key)

    app.state.server = server
    app.state.feature_count = int(w_int.shape[0])
    app.state.classes = class_names

    logger.info(
        "Server initialized with %s features and classes=%s",
        app.state.feature_count,
        class_names,
    )


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
    if not hasattr(app.state, "server") or app.state.server is None:
        raise HTTPException(status_code=503, detail="Server is not initialized")

    server = app.state.server
    if not isinstance(server, Server):
        raise HTTPException(status_code=500, detail="Invalid server state")

    if request.feature_count != len(request.encrypted_features):
        raise HTTPException(status_code=400, detail="feature_count does not match payload")

    if request.feature_count != int(app.state.feature_count):
        raise HTTPException(status_code=400, detail="feature_count does not match model")

    try:
        public_key = paillier.PaillierPublicKey(n=int(request.public_key_n))
        encrypted_features = [
            deserialize_ciphertext(public_key=public_key, ciphertext_str=ciphertext)
            for ciphertext in request.encrypted_features
        ]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid encrypted payload: {exc}") from exc

    server.public_key = public_key
    server.enc_b = public_key.encrypt(server.b_int)

    start = time.perf_counter()
    encrypted_score = server.compute_encrypted_score(enc_x=encrypted_features)
    compute_ms = (time.perf_counter() - start) * 1000.0

    return EncryptedInferResponse(
        encrypted_score=serialize_ciphertext(encrypted_number=encrypted_score),
        server_compute_ms=compute_ms,
        feature_count=request.feature_count,
    )
