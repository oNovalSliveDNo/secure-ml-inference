from collections.abc import Generator
from pathlib import Path

import phe as paillier
import pytest
from fastapi.testclient import TestClient

from api.main import app
from app.config import SCALE
from app.crypto import serialize_ciphertext

MODEL_PATH = Path("results/models/model.pkl")

if not MODEL_PATH.exists():
    pytest.skip("Model artifact is missing: results/models/model.pkl", allow_module_level=True)


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Фикстура с контекстным менеджером, чтобы отработал startup."""
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_model_info(client: TestClient) -> None:
    resp = client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "feature_count" in data
    assert "classes" in data
    assert isinstance(data["classes"], list)


def test_encrypted_inference(client: TestClient) -> None:
    info_resp = client.get("/model/info")
    assert info_resp.status_code == 200
    feature_count = info_resp.json()["feature_count"]

    public_key, _private_key = paillier.generate_paillier_keypair(n_length=512)

    encrypted_features = [serialize_ciphertext(public_key.encrypt(0)) for _ in range(feature_count)]

    payload = {
        "public_key_n": str(public_key.n),
        "encrypted_features": encrypted_features,
        "scale": SCALE,
        "feature_count": feature_count,
    }

    resp = client.post("/infer/encrypted", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "encrypted_score" in data
    assert "server_compute_ms" in data
    assert data["feature_count"] == feature_count
