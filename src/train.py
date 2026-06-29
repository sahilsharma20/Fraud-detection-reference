"""End-to-end training orchestrator: one command runs every stage in order.

    ingest -> EDA -> fraud models (+MLflow) -> cost-optimal threshold
           -> severity models (+MLflow) -> SHAP artifacts -> training summary

This is the reproducible spine of the project: `make train` (or
`python -m src.train`) regenerates every model, metric, plot and the runtime
inference config from the raw data, deterministically (fixed seeds).
"""

from __future__ import annotations

import json

import joblib

from monitoring.drift_monitor import build_reference_profile
from src.config import load_config, read_inference_config
from src.data_ingestion import ingest
from src.eda import run_eda
from src.explainability import generate_shap_artifacts
from src.fraud_model import train_fraud_models
from src.logger import get_logger
from src.severity_model import train_severity_models
from src.threshold_optimizer import optimize_threshold

log = get_logger(__name__)


def main() -> None:
    """Run the full training pipeline and write a consolidated summary."""
    cfg = load_config()
    log.info("=" * 70)
    log.info("TRAINING PIPELINE START")
    log.info("=" * 70)

    # 1. ingest + validate + split
    train_df, test_df = ingest(cfg)

    # 2. EDA (train fold only)
    eda_summary = run_eda(train_df, cfg)

    # 2b. snapshot the training distribution as the drift-monitoring reference
    build_reference_profile(train_df, cfg)

    # 3. Stage 1 — fraud classification (tracked in MLflow + registered)
    fraud_out = train_fraud_models(train_df, test_df, cfg)

    # 4. cost-sensitive threshold (tuned on held-out predictions)
    threshold_out = optimize_threshold(fraud_out["y_test"], fraud_out["proba_test"], cfg)

    # 5. Stage 2 — severity regression on GENUINE claims only (leakage guard)
    severity_out = train_severity_models(train_df, test_df, cfg)

    # 6. SHAP artifacts for the report (load the persisted best pipelines)
    fraud_pipe = joblib.load(cfg.path("paths.fraud_model"))
    severity_pipe = joblib.load(cfg.path("paths.severity_model"))
    fraud_col = cfg.get("data.target_fraud")
    generate_shap_artifacts(fraud_pipe, test_df.drop(columns=[fraud_col]), "fraud", cfg)
    genuine_test = test_df[test_df[fraud_col] != cfg.get("data.fraud_positive_label")]
    generate_shap_artifacts(
        severity_pipe,
        genuine_test.drop(columns=[fraud_col, cfg.get("data.target_severity")]),
        "severity", cfg,
    )

    # 7. consolidated training summary (handy for the report builder)
    summary = {
        "eda": eda_summary,
        "fraud": {"best_model": fraud_out["best_model"],
                  "registry_version": fraud_out["registry_version"],
                  "metrics": fraud_out["results"]},
        "threshold": threshold_out,
        "severity": {"best_model": severity_out["best_model"],
                     "registry_version": severity_out["registry_version"],
                     "metrics": severity_out["results"],
                     "business_statement": severity_out["business_statement"]},
        "inference_config": read_inference_config(cfg),
    }
    with open(cfg.path("paths.metrics_dir") / "training_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    log.info("=" * 70)
    log.info("TRAINING COMPLETE — fraud=%s severity=%s threshold=%.3f",
             fraud_out["best_model"], severity_out["best_model"],
             threshold_out["chosen_threshold"])
    log.info("=" * 70)


if __name__ == "__main__":
    main()
