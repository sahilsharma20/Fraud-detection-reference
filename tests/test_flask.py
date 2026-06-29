"""Integration tests for the Flask serving layer (flask_app.py).

Flask is the primary deployed app, so we smoke-test the same chained pipeline
through Flask's test client. Skipped if the trained models aren't present.
"""

from __future__ import annotations

import pytest

from src.config import load_config
from src.inference_pipeline import sample_claim

_cfg = load_config()
_MODELS_PRESENT = _cfg.path("paths.fraud_model").exists() and _cfg.path("paths.severity_model").exists()
pytestmark = pytest.mark.skipif(not _MODELS_PRESENT, reason="run `make train` first")


@pytest.fixture(scope="module")
def client():
    from flask_app import app

    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c


def test_flask_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["models_loaded"] is True


def test_flask_genuine_returns_severity(client):
    r = client.post("/predict", json=sample_claim(genuine=True))
    body = r.get_json()
    assert body["fraud"]["verdict"] == "GENUINE"
    assert body["severity"]["predicted_amount"] is not None


def test_flask_fraud_suppresses_severity(client):
    r = client.post("/predict", json=sample_claim(genuine=False))
    body = r.get_json()
    assert body["fraud"]["verdict"] == "NEEDS_REVIEW"
    assert body["severity"].get("predicted_amount") is None


def test_flask_validation_422(client):
    bad = sample_claim(genuine=True)
    bad["age"] = 5  # below min=16
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
