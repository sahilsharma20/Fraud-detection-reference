"""Unit tests for data ingestion: schema validation + stratified split."""

from __future__ import annotations

import pandera as pa
import pytest

from src.data_ingestion import build_raw_schema, split_data
from src.exception import FraudDetectionError

_SCHEMA_ERR = (pa.errors.SchemaError, pa.errors.SchemaErrors)


def test_schema_accepts_valid_frame(small_df, cfg):
    schema = build_raw_schema(cfg)
    validated = schema.validate(small_df)
    assert len(validated) == len(small_df)


def test_schema_rejects_bad_fraud_label(small_df, cfg):
    bad = small_df.copy()
    bad.loc[0, "fraud_reported"] = "MAYBE"  # not in {Y, N}
    schema = build_raw_schema(cfg)
    with pytest.raises(_SCHEMA_ERR):  # pandera raises SchemaError/SchemaErrors
        schema.validate(bad, lazy=True)


def test_schema_rejects_negative_target(small_df, cfg):
    bad = small_df.copy()
    bad.loc[0, "total_claim_amount"] = -100
    schema = build_raw_schema(cfg)
    with pytest.raises(_SCHEMA_ERR):
        schema.validate(bad, lazy=True)


def test_split_preserves_fraud_rate_and_sizes(small_df, cfg):
    train_df, test_df = split_data(small_df, cfg)
    assert len(train_df) + len(test_df) == len(small_df)
    # stratification keeps the fraud rate close across folds (within 5 pts)
    r_all = (small_df["fraud_reported"] == "Y").mean()
    r_test = (test_df["fraud_reported"] == "Y").mean()
    assert abs(r_all - r_test) < 0.05


def test_load_raw_missing_file_raises(tmp_path, cfg, monkeypatch):
    from src import data_ingestion

    # point at a non-existent path -> must raise our custom exception
    monkeypatch.setattr(cfg, "path", lambda key: tmp_path / "nope.csv")
    with pytest.raises(FraudDetectionError):
        data_ingestion.load_raw_data(cfg)
