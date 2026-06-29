"""Module 1 — Data ingestion: load, validate at the boundary, and split.

DESIGN DECISIONS
----------------
1. Schema validation with **pandera** AT THE DATA BOUNDARY.
   Why: the #1 cause of silent ML bugs is bad data sailing straight into the
   model — a renamed column, a string where a number should be, an out-of-range
   value. We validate the raw frame the moment it's read. If the contract is
   violated we raise immediately with a precise message, instead of producing a
   confidently-wrong model three stages later. (Pydantic guards the *API*
   boundary in app.py; pandera guards the *training-data* boundary here — right
   tool for each boundary.)

2. **Random stratified** split, NOT time-based — and here's the justification a
   fresher skips:
   A time-based split is correct only when (a) rows have a trustworthy temporal
   order AND (b) production use is "train on past, predict future". This dataset
   has an `incident_date`, but claims are not a forecasting problem — we score
   each claim independently as it arrives, and the fraud mechanism is not a
   time series. So a random split is appropriate. We DO stratify on the fraud
   label so the ~25% prevalence is identical in train and test — without
   stratification a small test fold can swing the fraud rate by several points
   and make metrics noisy/misleading.

FRESHER PITFALL AVOIDED
-----------------------
   Splitting AFTER fitting transformers (or fitting on the full frame) leaks
   test information into training. We split FIRST, here, and every transformer
   downstream is fit on train only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema
from sklearn.model_selection import train_test_split

from src.config import Config, load_config
from src.exception import FraudDetectionError
from src.logger import get_logger

log = get_logger(__name__)


def build_raw_schema(cfg: Config) -> DataFrameSchema:
    """Construct the pandera contract for the RAW Kaggle claims file.

    We validate the columns the pipeline actually depends on (presence, dtype,
    domain) and tolerate extras (``strict=False``) like the junk ``_c39`` column.

    Args:
        cfg: Loaded project config (supplies target/label names).

    Returns:
        A pandera :class:`DataFrameSchema`.

    """
    fraud_col = cfg.get("data.target_fraud")
    sev_col = cfg.get("data.target_severity")

    # nullable=True on '?'-bearing categoricals because the raw file uses '?'
    # as an in-band missing marker; we let the imputer handle them downstream.
    return DataFrameSchema(
        {
            "months_as_customer": Column(int, Check.ge(0), coerce=True),
            "age": Column(int, Check.in_range(15, 120), coerce=True),
            "policy_annual_premium": Column(float, Check.gt(0), coerce=True, nullable=True),
            "incident_hour_of_the_day": Column(int, Check.in_range(0, 23), coerce=True),
            "number_of_vehicles_involved": Column(int, Check.ge(0), coerce=True),
            "incident_severity": Column(str, nullable=True),
            "insured_sex": Column(str, Check.isin(["MALE", "FEMALE"]), nullable=True),
            # The targets MUST be present and valid — these are non-negotiable.
            sev_col: Column(int, Check.ge(0), coerce=True),
            fraud_col: Column(str, Check.isin(["Y", "N"])),
        },
        strict=False,       # extra columns (e.g. _c39) are allowed
        coerce=True,
        name="raw_insurance_claims",
    )


def load_raw_data(cfg: Config | None = None) -> pd.DataFrame:
    """Read the raw CSV and validate it against the schema contract.

    Args:
        cfg: Optional config; loaded from ``config.yaml`` if not supplied.

    Returns:
        The validated raw DataFrame.

    Raises:
        FraudDetectionError: If the file is missing or fails schema validation.

    """
    cfg = cfg or load_config()
    raw_path: Path = cfg.path("paths.raw_data")
    if not raw_path.exists():
        raise FraudDetectionError(
            f"Raw data not found at {raw_path}. Run `make data` to generate the "
            f"synthetic dataset, or drop the Kaggle insurance_claims.csv there."
        )

    log.info("Loading raw data from %s", raw_path)
    try:
        df = pd.read_csv(raw_path)
    except Exception as exc:  # noqa: BLE001
        raise FraudDetectionError(exc) from exc

    schema = build_raw_schema(cfg)
    try:
        df = schema.validate(df, lazy=True)  # lazy=collect ALL errors, not just first
    except pa.errors.SchemaErrors as exc:
        # Surface a compact, actionable summary instead of a wall of stack trace.
        log.error("Schema validation failed:\n%s", exc.failure_cases.head(20))
        raise FraudDetectionError(f"Raw data failed schema validation: {exc}") from exc

    log.info("Raw data validated OK — shape=%s", df.shape)
    return df


def split_data(
    df: pd.DataFrame, cfg: Config | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified random train/test split (see module docstring for rationale).

    Args:
        df: Validated raw frame.
        cfg: Optional config.

    Returns:
        ``(train_df, test_df)``.

    """
    cfg = cfg or load_config()
    fraud_col = cfg.get("data.target_fraud")
    test_size = float(cfg.get("data.test_size"))
    seed = int(cfg.get("project.random_state"))

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[fraud_col],  # preserve fraud prevalence in both folds
    )
    log.info(
        "Split -> train=%d (%.1f%% fraud)  test=%d (%.1f%% fraud)",
        len(train_df), (train_df[fraud_col] == "Y").mean() * 100,
        len(test_df), (test_df[fraud_col] == "Y").mean() * 100,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def ingest(cfg: Config | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end ingestion: load + validate + split, persisting both folds.

    Returns:
        ``(train_df, test_df)``.

    """
    cfg = cfg or load_config()
    df = load_raw_data(cfg)
    train_df, test_df = split_data(df, cfg)

    cfg.path("paths.processed_dir").mkdir(parents=True, exist_ok=True)
    train_df.to_csv(cfg.path("paths.train_data"), index=False)
    test_df.to_csv(cfg.path("paths.test_data"), index=False)
    log.info("Persisted train/test to %s", cfg.path("paths.processed_dir"))
    return train_df, test_df


if __name__ == "__main__":
    ingest()
