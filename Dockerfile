# ═══════════════════════════════════════════════════════════════════════════════
# Multi-stage build (Module 10).
#  * Stage 1 "builder": installs runtime deps into a venv (kept separate so build
#    tools never leak into the final image).
#  * Stage 2 "runtime": copies just the venv + app code + pre-trained models onto
#    a slim base, runs as a NON-ROOT user.
# Only requirements-serve.txt is installed (no mlflow/matplotlib/pytest) -> small,
# fast-booting image.
# ═══════════════════════════════════════════════════════════════════════════════

# ---------- Stage 1: builder ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

# build-essential needed transiently for any wheels that compile (e.g. shap deps)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated venv we can copy wholesale into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

# libgomp1 is XGBoost's OpenMP runtime dependency (needed to load the model).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000
WORKDIR /app

# bring the prebuilt venv from the builder
COPY --from=builder /opt/venv /opt/venv

# copy ONLY what the service needs at runtime (see .dockerignore for exclusions)
COPY src/ ./src/
COPY flask_app.py config.yaml ./
COPY frontend/ ./frontend/
COPY static/ ./static/
COPY models/ ./models/

USER appuser
EXPOSE 8000

# Serve the Flask app with gunicorn (Linux). Honour the platform-provided $PORT
# (Render/Railway inject it); default 8000 locally.
#   --workers 1 + --preload: load the models/ML libs ONCE (fits the 512MB free tier;
#                            2+ workers each re-load ~250MB and would OOM).
#   --threads 4:             concurrency for the SHAP-bound requests within one worker.
CMD ["sh", "-c", "gunicorn flask_app:app -b 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 120 --preload"]

# Container-level health check hitting the readiness endpoint.
HEALTHCHECK --interval=30s --timeout=4s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,os,sys; sys.exit(0 if 'ok' in urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health').read().decode() else 1)"
