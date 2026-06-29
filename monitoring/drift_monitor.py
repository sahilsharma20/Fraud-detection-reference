"""Module 11 — Data-drift monitoring (PSI) + retraining policy.

WHY THIS IS A FRESHER DIFFERENTIATOR
------------------------------------
Most portfolio projects stop at "model trained, API deployed" and silently rot:
the world shifts (new fraud tactics, inflation lifting claim amounts, a new
region onboarded) and yesterday's model quietly degrades with NOBODY watching.
A production engineer ships the monitoring on day one.

We compute the **Population Stability Index (PSI)** between the training
distribution (the "reference profile" snapshotted at train time) and a batch of
incoming claims, per feature. PSI is the industry standard for tabular drift —
no heavy dependency required, it's ~15 lines of maths:

    PSI = Σ (actual%% - expected%%) * ln(actual%% / expected%%)

Rule of thumb (encoded in config):
    PSI < 0.10   -> stable, no action
    0.10–0.25    -> moderate drift, investigate / schedule a refresh
    PSI > 0.25   -> significant drift, RETRAIN

RETRAINING PLAN (the "how", documented; see also the report)
------------------------------------------------------------
Triggers (any one fires a retrain):
  1. SCHEDULED: monthly cron retrain on the latest labelled window (claims have a
     resolution lag, so labels mature ~60-90 days after the incident).
  2. DRIFT-BASED: this monitor runs nightly on the day's claims; if any key
     feature's PSI > 0.25 (or aggregate PSI > 0.25) it opens a retrain ticket.
  3. PERFORMANCE-BASED: once labels mature, if rolling PR-AUC drops > 5 points
     vs the registered model's validation score, retrain.
Mechanics: retrain -> evaluate vs current registered model on a frozen holdout ->
promote in the MLflow Model Registry ONLY if it beats production (champion/
challenger) -> canary 10%% of traffic -> full rollout. Never auto-promote blindly.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.feature_engineering import RawFeatureEngineer
from src.logger import get_logger

log = get_logger(__name__)

_EPS = 1e-6  # avoids log(0) / divide-by-zero in PSI when a bin empties out


def _apply_raw_fe(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Run the same raw cleaning the model sees, so drift is measured on model inputs."""
    fe = RawFeatureEngineer(
        bind_date_col=cfg.get("data.date_columns.bind_date"),
        incident_date_col=cfg.get("data.date_columns.incident_date"),
        drop_columns=list(cfg.get("data.drop_columns")),
    )
    return fe.transform(df)


def build_reference_profile(train_df: pd.DataFrame, cfg: Config | None = None) -> dict:
    """Snapshot the training distribution as the drift reference, and persist it.

    For numeric features we store decile bin edges + bin proportions; for
    categoricals we store category proportions.

    Args:
        train_df: The training fold (the distribution the model learned).
        cfg: Optional config.

    Returns:
        The reference profile dict (also written to paths.reference_profile).

    """
    cfg = cfg or load_config()
    n_bins = int(cfg.get("monitoring.numeric_bins"))
    df = _apply_raw_fe(train_df, cfg)

    numeric = [c for c in cfg.get("features.numeric") if c in df.columns]
    categorical = [c for c in cfg.get("features.categorical") if c in df.columns]

    profile: dict = {"numeric": {}, "categorical": {}, "n_reference": int(len(df))}

    for col in numeric:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        # quantile edges; unique() collapses ties so monotonic bins are guaranteed
        edges = np.unique(np.quantile(series, np.linspace(0, 1, n_bins + 1)))
        counts, _ = np.histogram(series, bins=edges)
        props = (counts / max(counts.sum(), 1)).tolist()
        profile["numeric"][col] = {"edges": edges.tolist(), "props": props}

    for col in categorical:
        freq = df[col].astype("object").fillna("__missing__").value_counts(normalize=True)
        profile["categorical"][col] = freq.to_dict()

    out = cfg.path("paths.reference_profile")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2)
    log.info("Reference profile written: %d numeric + %d categorical features -> %s",
             len(numeric), len(categorical), out)
    return profile


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population Stability Index between two proportion vectors."""
    expected = np.clip(expected, _EPS, None)
    actual = np.clip(actual, _EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _numeric_psi(ref: dict, series: pd.Series) -> float:
    edges = np.array(ref["edges"])
    expected = np.array(ref["props"])
    counts, _ = np.histogram(pd.to_numeric(series, errors="coerce").dropna(), bins=edges)
    actual = counts / max(counts.sum(), 1)
    return _psi(expected, actual)


def _categorical_psi(ref: dict, series: pd.Series) -> float:
    actual_freq = series.astype("object").fillna("__missing__").value_counts(normalize=True).to_dict()
    cats = set(ref) | set(actual_freq)
    expected = np.array([ref.get(c, 0.0) for c in cats])
    actual = np.array([actual_freq.get(c, 0.0) for c in cats])
    return _psi(expected, actual)


def compute_drift(current_df: pd.DataFrame, cfg: Config | None = None) -> dict:
    """Compute per-feature PSI of incoming data vs the training reference.

    Args:
        current_df: A batch of incoming (raw) claims to check for drift.
        cfg: Optional config.

    Returns:
        A report dict: per-feature PSI, max PSI, status, drifted features and a
        plain-English recommendation.

    """
    cfg = cfg or load_config()
    ref_path = cfg.path("paths.reference_profile")
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference profile not found at {ref_path}. Run training first "
            "(build_reference_profile runs inside `make train`)."
        )
    with open(ref_path, encoding="utf-8") as fh:
        profile = json.load(fh)

    df = _apply_raw_fe(current_df, cfg)
    warn = float(cfg.get("monitoring.psi_warn"))
    alert = float(cfg.get("monitoring.psi_alert"))

    per_feature: dict[str, float] = {}
    for col, ref in profile["numeric"].items():
        if col in df.columns:
            per_feature[col] = round(_numeric_psi(ref, df[col]), 4)
    for col, ref in profile["categorical"].items():
        if col in df.columns:
            per_feature[col] = round(_categorical_psi(ref, df[col]), 4)

    max_psi = max(per_feature.values(), default=0.0)
    drifted = {k: v for k, v in per_feature.items() if v >= warn}

    if max_psi >= alert:
        status, rec = "ALERT", "Significant drift (PSI ≥ 0.25). Trigger a retrain."
    elif max_psi >= warn:
        status, rec = "WARN", "Moderate drift (PSI ≥ 0.10). Investigate; schedule a refresh."
    else:
        status, rec = "STABLE", "No material drift. No action needed."

    report = {
        "status": status, "max_psi": round(max_psi, 4),
        "thresholds": {"warn": warn, "alert": alert},
        "drifted_features": drifted, "per_feature_psi": per_feature,
        "n_current": int(len(current_df)), "recommendation": rec,
    }
    log.info("Drift status=%s max_psi=%.3f drifted=%d/%d -> %s",
             status, max_psi, len(drifted), len(per_feature), rec)
    return report


if __name__ == "__main__":
    # Demo: compare the test fold (no drift expected) and a deliberately shifted
    # batch (inflated claim amounts + worse severities) to show the monitor firing.
    cfg = load_config()
    test_df = pd.read_csv(cfg.path("paths.test_data"))
    print("\n--- test fold (expect STABLE) ---")
    compute_drift(test_df, cfg)

    shifted = test_df.copy()
    shifted["total_claim_amount"] = shifted["total_claim_amount"] * 2.5
    shifted["incident_severity"] = "Total Loss"
    shifted["age"] = shifted["age"].clip(upper=25)
    print("\n--- shifted batch (expect WARN/ALERT) ---")
    report = compute_drift(shifted, cfg)
    print(json.dumps(report, indent=2))
