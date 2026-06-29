# ═══════════════════════════════════════════════════════════════════════════════
# Common commands as a task runner. `make help` lists everything.
# Works on Windows (Git Bash / WSL) and Unix. Uses the venv python if present.
# ═══════════════════════════════════════════════════════════════════════════════
.DEFAULT_GOAL := help
PY := python

ifeq ($(OS),Windows_NT)
	VENV_PY := .venv/Scripts/python.exe
else
	VENV_PY := .venv/bin/python
endif
ifneq ("$(wildcard $(VENV_PY))","")
	PY := $(VENV_PY)
endif

.PHONY: help install data train report serve docker-build docker-run test lint format clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create venv-friendly install of all dependencies
	$(PY) -m pip install -r requirements.txt

data:  ## Generate the synthetic dataset (schema-identical to the Kaggle file)
	$(PY) -m scripts.generate_synthetic_data

train:  ## Run the full pipeline: ingest -> EDA -> fraud -> threshold -> severity -> SHAP
	$(PY) -m src.train

report:  ## Build the self-contained shareable report.html
	$(PY) -m scripts.build_report

serve:  ## Run the FastAPI app + web UI locally on :8000
	$(PY) -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

docker-build:  ## Build the production Docker image
	docker build -t fraud-severity:latest .

docker-run:  ## Run the container locally on :8000
	docker run --rm -p 8000:8000 fraud-severity:latest

test:  ## Run unit + integration tests
	$(PY) -m pytest -q

lint:  ## Lint with ruff
	$(PY) -m ruff check src tests app.py scripts monitoring

format:  ## Auto-format with ruff
	$(PY) -m ruff format src tests app.py scripts monitoring

clean:  ## Remove caches and generated artifacts
	rm -rf __pycache__ .pytest_cache .ruff_cache reports/plots reports/metrics logs
