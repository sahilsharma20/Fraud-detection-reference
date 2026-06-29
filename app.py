"""FastAPI service (Module 9) — serves BOTH the JSON API and the web UI.

ONE deployable unit, ONE source of truth: every prediction (API or browser) goes
through ``src.inference_pipeline.get_service().predict`` — the UI never
re-implements the chaining. Serving the static frontend from the same app means a
single Render/Railway service and no CORS dance.

Endpoints:
    GET  /            -> the web UI (Deliverable B)
    POST /predict     -> chained fraud + conditional severity (Pydantic-validated)
    GET  /sample      -> a prefilled example claim (genuine|fraud) for the form
    GET  /health      -> liveness probe (models loaded?)
    GET  /docs        -> auto Swagger UI
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import load_config
from src.exception import FraudDetectionError
from src.inference_pipeline import get_service, sample_claim
from src.logger import get_logger
from src.schemas import ClaimRequest, PredictionResponse

log = get_logger(__name__)
ROOT = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the models once at startup so the first request isn't slow."""
    try:
        get_service()  # loads models + builds SHAP explainers
        log.info("Models warmed and ready.")
    except FraudDetectionError as exc:
        # Don't crash the server — /health will report not-ready and /predict 503s.
        log.error("Startup model load failed: %s", exc)
    yield


app = FastAPI(
    title="Insurance Claim Fraud + Severity API",
    description="Two-stage system: Stage 1 flags fraud; genuine claims get a Stage 2 severity estimate.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_latency(request: Request, call_next):
    """Log per-request latency (an SRE-friendly habit freshers skip)."""
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-ms"] = f"{dt:.1f}"
    if request.url.path == "/predict":
        log.info("%s %s -> %s in %.1f ms", request.method, request.url.path,
                 response.status_code, dt)
    return response


@app.post("/predict", response_model=PredictionResponse)
async def predict(claim: ClaimRequest) -> PredictionResponse:
    """Score one claim through the chained two-stage pipeline."""
    try:
        result = get_service().predict(claim.to_claim_dict())
        return PredictionResponse(**result)
    except FraudDetectionError as exc:
        log.error("Prediction failed: %s", exc)
        return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/sample")
async def get_sample(type: str = "genuine") -> dict:
    """Return a prefilled example claim for the UI's quick-fill buttons."""
    return sample_claim(genuine=(type != "fraud"))


@app.get("/health")
async def health() -> dict:
    """Liveness + readiness probe."""
    try:
        svc = get_service()
        ready = svc.fraud_pipeline is not None and svc.severity_pipeline is not None
    except FraudDetectionError:
        ready = False
    return {"status": "ok" if ready else "degraded", "models_loaded": ready}


@app.get("/config")
async def config_summary() -> dict:
    """Expose non-sensitive runtime config the UI displays (threshold, currency)."""
    svc = get_service()
    cfg = load_config()
    return {
        "threshold": svc.threshold,
        "severity_mae": svc.severity_mae,
        "currency": cfg.get("business.currency_symbol"),
    }


# ── static assets + UI (mounted last so API routes take precedence) ──
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the single-page web UI."""
    return FileResponse(ROOT / "frontend" / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
