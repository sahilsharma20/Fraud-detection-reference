"""Unit tests for the feature-engineering pipeline + the leakage guard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import (
    RawFeatureEngineer,
    build_preprocessor,
    get_feature_names,
)


def test_raw_fe_derives_tenure_and_drops_columns(cfg):
    df = pd.DataFrame({
        "policy_bind_date": ["2010-01-01"],
        "incident_date": ["2012-01-01"],
        "policy_number": [12345],
        "incident_severity": ["?"],   # in-band missing marker
        "age": [40],
    })
    fe = RawFeatureEngineer(
        bind_date_col="policy_bind_date", incident_date_col="incident_date",
        drop_columns=["policy_number"],
    )
    out = fe.transform(df)
    assert "customer_tenure_days" in out.columns
    assert out["customer_tenure_days"].iloc[0] == 730  # 2010-01-01 -> 2012-01-01 = 730 days
    assert "policy_bind_date" not in out.columns and "incident_date" not in out.columns
    assert "policy_number" not in out.columns
    assert pd.isna(out["incident_severity"].iloc[0])    # '?' normalised to NaN


def test_severity_preprocessor_excludes_leakage_columns(small_df, cfg):
    """The claim-component columns must NOT appear in the severity feature matrix."""
    pre = build_preprocessor("severity", cfg)
    X = small_df.drop(columns=[cfg.get("data.target_severity"), cfg.get("data.target_fraud")])
    pre.fit(X)
    names = " ".join(get_feature_names(pre))
    for leak in cfg.get("data.severity_leakage_columns"):
        assert leak not in names, f"leakage column {leak} leaked into severity features"


def test_fraud_preprocessor_includes_claim_components(small_df, cfg):
    pre = build_preprocessor("fraud", cfg)
    X = small_df.drop(columns=[cfg.get("data.target_fraud")])
    pre.fit(X)
    names = get_feature_names(pre)
    assert "total_claim_amount" in names  # valid feature for the fraud model


def test_preprocessor_output_is_finite(small_df, cfg):
    pre = build_preprocessor("fraud", cfg)
    X = small_df.drop(columns=[cfg.get("data.target_fraud")])
    Xt = pre.fit_transform(X)
    assert np.isfinite(Xt).all()      # imputation leaves no NaNs/inf
    assert Xt.shape[0] == len(small_df)
