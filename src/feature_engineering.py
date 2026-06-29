"""Module 3 — Feature engineering as a persisted sklearn Pipeline.

DESIGN DECISION: wrap ALL transformations in a single sklearn ``Pipeline``
(``RawFeatureEngineer`` -> ``ColumnTransformer``) and persist it fitted. The same
object is used at train time AND inside the inference pipeline. This is the only
robust way to guarantee training/serving parity — there is literally no second
copy of the transformation logic to drift out of sync.

FRESHER PITFALLS AVOIDED
------------------------
* Fitting the scaler/encoder on the full dataset (or on test) -> leakage. Here
  the pipeline is ``fit`` on train only, then ``transform`` is applied to test
  and to live requests.
* ``pd.get_dummies`` at train and again at predict -> column mismatch when a
  category is unseen at inference. We use ``OneHotEncoder(handle_unknown=
  "ignore")`` so unseen categories degrade gracefully instead of crashing.
* Two feature sets for two models handled by NOT-selecting leakage columns in
  the ColumnTransformer (remainder="drop"), so the severity model can never see
  the claim-component columns that compose its target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import Config, load_config
from src.logger import get_logger

log = get_logger(__name__)

# The raw file encodes "unknown" as the literal string '?' in several columns.
# We normalise these to real NaNs so the imputer can handle them uniformly.
_MISSING_TOKENS = ["?", "NONE", "None", "none", ""]


class RawFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless raw-frame cleaning + date feature engineering.

    Runs FIRST in the pipeline. Responsibilities:
      * derive ``customer_tenure_days`` = incident_date - policy_bind_date,
      * drop identifier / free-text / raw-date / junk columns,
      * normalise in-band '?' missing markers to NaN.

    It is intentionally stateless (``fit`` is a no-op) so it is trivially
    consistent between train and inference.
    """

    def __init__(self, bind_date_col: str, incident_date_col: str, drop_columns: list[str]):
        self.bind_date_col = bind_date_col
        self.incident_date_col = incident_date_col
        self.drop_columns = drop_columns

    def fit(self, X: pd.DataFrame, y=None) -> RawFeatureEngineer:  # noqa: D102, N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: D102, N803
        df = X.copy()

        # ── engineered feature: customer tenure (days) at time of incident ──
        if self.bind_date_col in df and self.incident_date_col in df:
            bind = pd.to_datetime(df[self.bind_date_col], errors="coerce")
            incident = pd.to_datetime(df[self.incident_date_col], errors="coerce")
            tenure = (incident - bind).dt.days
            # clip negatives (data-entry errors) to 0; fill unknowns with median-ish 0
            df["customer_tenure_days"] = tenure.clip(lower=0).fillna(0).astype(float)

        # drop raw dates + configured identifier/junk columns (ignore if absent
        # so the same transformer works on a single-row inference payload too)
        to_drop = set(self.drop_columns) | {self.bind_date_col, self.incident_date_col}
        df = df.drop(columns=[c for c in to_drop if c in df.columns])

        # normalise in-band missing markers in object columns to np.nan.
        # NOTE: np.nan (not pd.NA) — pd.NA raises "boolean value of NA is
        # ambiguous" deep inside sklearn's imputer/encoder. np.nan is the dtype
        # sklearn expects for missing values.
        obj_cols = df.select_dtypes(include="object").columns
        if len(obj_cols):
            # option_context opts into pandas' future no-silent-downcast behaviour,
            # silencing the deprecation warning while keeping object dtype intact.
            with pd.option_context("future.no_silent_downcasting", True):
                df[obj_cols] = df[obj_cols].replace(_MISSING_TOKENS, np.nan)
        return df


def _resolve_feature_lists(cfg: Config, stage: str) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) for the given model stage.

    For ``severity`` the claim-component columns are deliberately excluded — they
    sum to the target and would be leakage (see config.severity_leakage_columns).
    """
    numeric = list(cfg.get("features.numeric"))
    categorical = list(cfg.get("features.categorical"))
    if stage == "fraud":
        numeric = numeric + list(cfg.get("features.numeric_claim_components"))
    elif stage == "severity":
        pass  # claim components intentionally NOT added -> not selected -> dropped
    else:
        raise ValueError(f"Unknown stage '{stage}' (expected 'fraud' or 'severity')")
    return numeric, categorical


def build_preprocessor(stage: str, cfg: Config | None = None) -> Pipeline:
    """Build an UNFITTED preprocessing pipeline for the given stage.

    Args:
        stage: ``"fraud"`` or ``"severity"`` — controls which columns are used.
        cfg: Optional config.

    Returns:
        An sklearn ``Pipeline``: RawFeatureEngineer -> ColumnTransformer.

    """
    cfg = cfg or load_config()
    numeric, categorical = _resolve_feature_lists(cfg, stage)

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),  # harmless for trees, required for LR/linear
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            # sparse_output=False so SHAP gets a dense matrix; ignore unseen levels
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10)),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",  # anything not listed (incl. severity leakage cols) is dropped
        verbose_feature_names_out=False,
    )

    raw_fe = RawFeatureEngineer(
        bind_date_col=cfg.get("data.date_columns.bind_date"),
        incident_date_col=cfg.get("data.date_columns.incident_date"),
        drop_columns=list(cfg.get("data.drop_columns")),
    )

    pipe = Pipeline(steps=[("raw_fe", raw_fe), ("columns", column_transformer)])
    log.info(
        "Built '%s' preprocessor: %d numeric + %d categorical input columns",
        stage, len(numeric), len(categorical),
    )
    return pipe


def get_feature_names(preprocessor: Pipeline) -> list[str]:
    """Extract output feature names from a fitted preprocessing pipeline.

    Used by SHAP/explainability to map matrix columns back to human concepts.
    """
    return list(preprocessor.named_steps["columns"].get_feature_names_out())
