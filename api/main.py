# api/main.py
"""FastAPI application entrypoint for model serving."""

from fastapi import FastAPI
from app.schemas import EncryptedInferRequest, EncryptedInferResponse
from app.server import Server
from app.model import load_model, extract_linear_params
from app.encoding import encode_weights, encode_bias
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Secure ML Inference API")

# Placeholder for server instance (will be initialized on startup)
server: Server = None


@app.on_event("startup")
async def startup_event():
    """Load model and initialize server instance."""
    # TODO: load model, extract weights, encode, create Server instance
    global server
    # server = Server(...)
    logger.info("Server initialized.")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/model/info")
async def model_info():
    """Return model metadata (feature count, class labels)."""
    # TODO: return useful info
    return {"feature_count": 30, "classes": ["malignant", "benign"]}


@app.post("/infer/encrypted", response_model=EncryptedInferResponse)
async def infer_encrypted(request: EncryptedInferRequest):
    """
    Accept encrypted features, compute encrypted score, return it.
    """
    # TODO: deserialize request, compute encrypted score, measure time, return response
    raise NotImplementedError
