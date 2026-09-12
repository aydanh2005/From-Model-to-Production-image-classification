import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_prediction_contract():
    response = client.post("/predict", json={"pixels": [0] * 784})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["probabilities"]) == 10
    assert sum(payload["probabilities"].values()) == pytest.approx(1.0, abs=0.001)
    assert payload["confidence"] == max(payload["probabilities"].values())
    assert isinstance(payload["manual_review_required"], bool)


def test_rejects_wrong_pixel_count():
    response = client.post("/predict", json={"pixels": [0] * 10})
    assert response.status_code == 422


def test_rejects_pixel_outside_range():
    response = client.post("/predict", json={"pixels": [300] * 784})
    assert response.status_code == 422


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json()["model_version"] == "fashion-mnist-rf-v1"

