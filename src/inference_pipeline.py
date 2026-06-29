"""Module 9 — Chained inference: the SINGLE SOURCE OF TRUTH for prediction.

The API (app.py) and the web UI both call this and nothing else. There is
exactly one implementation of the business rule:

    claim --> Stage 1 fraud model --> P(fraud)
          --> if P >= cost-optimal threshold:  verdict = NEEDS REVIEW, STOP
                                               (NO severity prediction returned)
          --> else (genuine):                  verdict = GENUINE,
                                               run Stage 2 severity model
          --> attach human-readable SHAP reasons either way

WHY A SINGLETON SERVICE (fresher pitfall avoided):
    Loading the joblib models and building SHAP explainers costs ~100ms-1s. A
    fresher reloads them on every request (or, worse, re-implements the chaining
    in the UI separately from the API, so they drift). We load once, cache the
    models + explainers, and serve every request from the warm objects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import Config, load_config, read_inference_config
from src.exception import FraudDetectionError
from src.explainability import _build_explainer, explain_instance, unwrap
from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class InferenceService:
    """Holds the warm models/explainers and runs the chained prediction."""

    cfg: Config = field(default_factory=load_config)
    fraud_pipeline: object = None
    severity_pipeline: object = None
    threshold: float = 0.5
    severity_mae: float | None = None
    _fraud_explainer: object = None
    _severity_explainer: object = None
    _required_columns: list[str] = field(default_factory=list)

    def load(self) -> InferenceService:
        """Load models + runtime config; build & cache SHAP explainers once."""
        fraud_path = self.cfg.path("paths.fraud_model")
        sev_path = self.cfg.path("paths.severity_model")
        if not fraud_path.exists() or not sev_path.exists():
            raise FraudDetectionError(
                "Models not found. Run `make train` before serving "
                f"(looked for {fraud_path} and {sev_path})."
            )
        self.fraud_pipeline = joblib.load(fraud_path)
        self.severity_pipeline = joblib.load(sev_path)

        # runtime values produced by training (cost-optimal threshold, MAE band)
        rc = read_inference_config(self.cfg)
        self.threshold = float(rc.get("fraud_threshold",
                                      self.cfg.get("threshold.default_threshold")))
        self.severity_mae = rc.get("severity_mae")

        # warm the SHAP explainers (built from the fitted estimators)
        try:
            f_pre, f_model = unwrap(self.fraud_pipeline)
            self._fraud_explainer = _build_explainer(f_model, None)
            s_pre, s_model = unwrap(self.severity_pipeline)
            self._severity_explainer = _build_explainer(s_model, None)
        except Exception as exc:  # noqa: BLE001 - explainers are best-effort
            log.warning("Could not pre-build SHAP explainers: %s", exc)

        self._required_columns = self._compute_required_columns()
        log.info("InferenceService loaded. threshold=%.3f severity_mae=%s",
                 self.threshold, self.severity_mae)
        return self

    def _compute_required_columns(self) -> list[str]:
        """Columns the preprocessors expect to exist on the input frame."""
        cfg = self.cfg
        cols = set(cfg.get("features.numeric")) | set(cfg.get("features.categorical"))
        cols |= set(cfg.get("features.numeric_claim_components"))
        cols |= {cfg.get("data.date_columns.bind_date"),
                 cfg.get("data.date_columns.incident_date")}
        cols.discard("customer_tenure_days")  # derived inside the pipeline
        return sorted(cols)

    def _to_frame(self, claim: dict) -> pd.DataFrame:
        """Build a 1-row frame, ensuring every required raw column exists.

        Missing fields become NaN and are imputed inside the pipeline, so a
        partially-filled form still produces a prediction (with a wider implied
        uncertainty) rather than a 500 error.
        """
        df = pd.DataFrame([claim])
        for col in self._required_columns:
            if col not in df.columns:
                df[col] = np.nan
        return df

    def predict(self, claim: dict) -> dict:
        """Run the full chained prediction for one claim.

        Args:
            claim: Raw claim fields (a dict; missing fields tolerated).

        Returns:
            A structured result dict (see module docstring for the chaining rule).

        """
        if self.fraud_pipeline is None:
            self.load()
        t0 = time.perf_counter()
        row = self._to_frame(claim)

        # ── Stage 1: fraud ──
        fraud_prob = float(self.fraud_pipeline.predict_proba(row)[:, 1][0])
        flagged = fraud_prob >= self.threshold
        fraud_reasons = explain_instance(
            self.fraud_pipeline, row, stage="fraud", explainer=self._fraud_explainer, cfg=self.cfg
        )

        result: dict = {
            "fraud": {
                "verdict": "NEEDS_REVIEW" if flagged else "GENUINE",
                "is_fraud_suspected": flagged,
                "fraud_probability": round(fraud_prob, 4),
                "threshold": round(self.threshold, 4),
                "reasons": fraud_reasons,
            },
            "severity": None,
            "meta": {
                "fraud_model": type(unwrap(self.fraud_pipeline)[1]).__name__,
                "severity_model": type(unwrap(self.severity_pipeline)[1]).__name__,
                "currency": self.cfg.get("business.currency_symbol"),
            },
        }

        # ── Stage 2: severity — ONLY for genuine claims (the chaining rule) ──
        if flagged:
            result["severity"] = {
                "predicted_amount": None,
                "note": "Severity not predicted — claim flagged for manual review.",
            }
        else:
            amount = float(self.severity_pipeline.predict(row)[0])
            amount = max(amount, 0.0)
            mae = float(self.severity_mae) if self.severity_mae else None
            severity_reasons = explain_instance(
                self.severity_pipeline, row, stage="severity",
                explainer=self._severity_explainer, cfg=self.cfg,
            )
            result["severity"] = {
                "predicted_amount": round(amount, 2),
                "currency": self.cfg.get("business.currency_symbol"),
                "error_band": round(mae, 2) if mae else None,
                "lower_estimate": round(max(amount - mae, 0), 2) if mae else None,
                "upper_estimate": round(amount + mae, 2) if mae else None,
                "reasons": severity_reasons,
            }

        result["meta"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return result


@lru_cache(maxsize=1)
def get_service() -> InferenceService:
    """Process-wide singleton accessor (warm models shared across requests)."""
    return InferenceService().load()


def sample_claim(genuine: bool = True) -> dict:
    """A realistic example claim for the UI's 'load sample' button and tests."""
    base = {
        "months_as_customer": 120, "age": 42,
        "policy_bind_date": "2010-05-12", "policy_state": "OH", "policy_csl": "250/500",
        "policy_deductable": 1000, "policy_annual_premium": 1280.5, "umbrella_limit": 0,
        "insured_sex": "MALE", "insured_education_level": "Masters",
        "insured_occupation": "exec-managerial", "insured_hobbies": "reading",
        "insured_relationship": "husband", "capital-gains": 0, "capital-loss": 0,
        "incident_date": "2015-02-20", "incident_type": "Multi-vehicle Collision",
        "collision_type": "Rear Collision", "incident_severity": "Minor Damage",
        "authorities_contacted": "Police", "incident_state": "NY", "incident_city": "Columbus",
        "incident_hour_of_the_day": 14, "number_of_vehicles_involved": 2,
        "property_damage": "NO", "bodily_injuries": 0, "witnesses": 2,
        "police_report_available": "YES", "total_claim_amount": 32000,
        "injury_claim": 6000, "property_claim": 6000, "vehicle_claim": 20000,
        "auto_make": "Toyota", "auto_year": 2012,
    }
    if not genuine:
        # nudge the fields the synthetic signal keys on toward "suspicious"
        base.update({
            "incident_severity": "Major Damage", "police_report_available": "NO",
            "authorities_contacted": "None", "insured_hobbies": "chess",
            "total_claim_amount": 92000, "injury_claim": 22000,
            "property_claim": 20000, "vehicle_claim": 50000,
        })
    return base
