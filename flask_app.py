"""Flask serving layer — an alternative to app.py (FastAPI).

WHY THIS EXISTS:
    The project's serving logic lives entirely in ``src/inference_pipeline.py``
    (the single source of truth). This Flask app is a thin HTTP shell around it,
    proving the same chained pipeline can be served by *any* framework with zero
    duplicated prediction logic. It serves the IDENTICAL frontend (frontend/ +
    static/) and the same endpoints as the FastAPI version.

Run locally:
    python flask_app.py                 # http://localhost:5000
    # or, production WSGI:
    gunicorn flask_app:app -b 0.0.0.0:5000

Reuses for free:
    * src.inference_pipeline.get_service()  -> warm, cached models + SHAP
    * src.schemas.ClaimRequest              -> the SAME Pydantic validation
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from pydantic import ValidationError

from src.config import load_config
from src.exception import FraudDetectionError
from src.inference_pipeline import get_service, sample_claim
from src.logger import get_logger
from src.schemas import ClaimRequest

log = get_logger(__name__)
ROOT = Path(__file__).resolve().parent

# static_url_path="/static" matches the asset paths in frontend/index.html
app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")
CORS(app)  # harmless same-origin; lets the UI be hosted separately if ever needed

# Warm the models once at import time so the first request isn't slow (works under
# gunicorn too, where __main__ never runs).
try:
    get_service()
    log.info("Flask: models warmed and ready.")
except FraudDetectionError as exc:
    log.error("Flask startup model load failed: %s", exc)


@app.get("/")
def index():
    """Serve the single-page web UI."""
    return send_from_directory(ROOT / "frontend", "index.html")


@app.post("/predict")
def predict():
    """Score one claim through the chained two-stage pipeline (Pydantic-validated)."""
    try:
        payload = request.get_json(force=True, silent=True) or {}
        claim = ClaimRequest(**payload)           # same validation as the FastAPI app
    except ValidationError as exc:
        return jsonify({"detail": exc.errors()}), 422
    try:
        result = get_service().predict(claim.to_claim_dict())
        return jsonify(result)
    except FraudDetectionError as exc:
        log.error("Prediction failed: %s", exc)
        return jsonify({"detail": str(exc)}), 503


@app.get("/sample")
def get_sample():
    """Return a prefilled example claim for the UI's quick-fill buttons."""
    return jsonify(sample_claim(genuine=(request.args.get("type") != "fraud")))


@app.get("/config")
def config_summary():
    """Expose non-sensitive runtime config the UI displays."""
    svc = get_service()
    cfg = load_config()
    return jsonify({
        "threshold": svc.threshold,
        "severity_mae": svc.severity_mae,
        "currency": cfg.get("business.currency_symbol"),
    })


@app.get("/health")
def health():
    """Liveness + readiness probe."""
    try:
        svc = get_service()
        ready = svc.fraud_pipeline is not None and svc.severity_pipeline is not None
    except FraudDetectionError:
        ready = False
    return jsonify({"status": "ok" if ready else "degraded", "models_loaded": ready})


@app.after_request
def add_latency_header(response):
    """Mirror the inference latency to a response header (per-request timing)."""
    response.headers["X-Powered-By"] = "Flask + scikit-learn"
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
