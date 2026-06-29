"""Integration tests hitting the FastAPI /predict endpoint end-to-end.

These exercise the SAME chained inference path the web UI uses, so they catch
regressions in the contract the front-end depends on. Skipped automatically if
the trained models aren't present (they ARE committed, so CI runs them).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.config import load_config
from src.inference_pipeline import sample_claim

_cfg = load_config()
_MODELS_PRESENT = _cfg.path("paths.fraud_model").exists() and _cfg.path("paths.severity_model").exists()
pytestmark = pytest.mark.skipif(not _MODELS_PRESENT, reason="run `make train` first")


@pytest.fixture(scope="module")
def client():
    from app import app  # imported lazily so collection works without models

    with TestClient(app) as c:   # context manager triggers startup (model warmup)
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["models_loaded"] is True


def test_predict_genuine_path_returns_severity(client):
    r = client.post("/predict", json=sample_claim(genuine=True))
    assert r.status_code == 200
    body = r.json()
    assert body["fraud"]["verdict"] == "GENUINE"
    # genuine claims MUST get a severity estimate with an error band + reasons
    assert body["severity"]["predicted_amount"] is not None
    assert body["severity"]["error_band"] is not None
    assert len(body["fraud"]["reasons"]) >= 1


def test_predict_fraud_path_suppresses_severity(client):
    r = client.post("/predict", json=sample_claim(genuine=False))
    assert r.status_code == 200
    body = r.json()
    assert body["fraud"]["verdict"] == "NEEDS_REVIEW"
    # the chaining rule: NO severity number for a flagged claim
    assert body["severity"].get("predicted_amount") is None
    assert "review" in body["severity"]["note"].lower()


def test_predict_validation_rejects_bad_input(client):
    bad = sample_claim(genuine=True)
    bad["age"] = 5  # below the min=16 bound in the Pydantic schema
    r = client.post("/predict", json=bad)
    assert r.status_code == 422  # FastAPI validation error, never reaches the model


def test_predict_reports_latency(client):
    r = client.post("/predict", json=sample_claim(genuine=True))
    assert r.json()["meta"]["latency_ms"] >= 0
    assert "X-Process-Time-ms" in r.headers
