"""Module 8 — SHAP explainability for BOTH models.

Two jobs:
  1. ``generate_shap_artifacts`` — global beeswarm + a few local waterfalls saved
     for the report (run once at training time).
  2. ``explain_instance`` — fast, per-request, human-readable top factors wired
     into the web UI result card. This is what turns a black-box score into
     "flagged because: incident severity is 'Major Damage', no police report,
     claim amount is high" — the difference between a model nobody trusts and one
     a claims handler will actually act on.

FRESHER PITFALL AVOIDED: dumping raw one-hot feature codes
(``incident_severity_Major Damage = 0.34``) into the UI. We translate every
factor into plain English with the original value, and a direction
("increases / decreases risk").
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.config import Config, load_config  # noqa: E402
from src.feature_engineering import get_feature_names  # noqa: E402
from src.logger import get_logger  # noqa: E402

log = get_logger(__name__)

# Human labels for columns whose snake_case isn't self-explanatory enough.
_HUMAN_LABELS = {
    "customer_tenure_days": "Customer tenure (days)",
    "months_as_customer": "Months as customer",
    "policy_annual_premium": "Annual premium",
    "policy_deductable": "Policy deductible",
    "umbrella_limit": "Umbrella limit",
    "capital-gains": "Capital gains",
    "capital-loss": "Capital loss",
    "incident_hour_of_the_day": "Incident hour",
    "number_of_vehicles_involved": "Vehicles involved",
    "bodily_injuries": "Bodily injuries",
    "total_claim_amount": "Total claim amount",
    "injury_claim": "Injury claim",
    "property_claim": "Property claim",
    "vehicle_claim": "Vehicle claim",
    "auto_year": "Vehicle year",
    "incident_severity": "Incident severity",
    "police_report_available": "Police report available",
    "authorities_contacted": "Authorities contacted",
    "insured_hobbies": "Insured hobbies",
    "incident_type": "Incident type",
    "collision_type": "Collision type",
    "insured_occupation": "Occupation",
    "insured_education_level": "Education level",
}


def _pretty(col: str) -> str:
    return _HUMAN_LABELS.get(col, col.replace("_", " ").replace("-", " ").capitalize())


def unwrap(pipeline: Pipeline) -> tuple[object, object]:
    """Return ``(preprocessor, estimator)`` from a full pipeline.

    Transparently unwraps ``TransformedTargetRegressor`` so SHAP sees the actual
    fitted tree/linear model, not the target-transform wrapper.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    if isinstance(model, TransformedTargetRegressor):
        model = model.regressor_
    return preprocessor, model


def _is_tree(model) -> bool:
    return hasattr(model, "feature_importances_")


def _build_explainer(model, background: np.ndarray | None):
    """Pick the right SHAP explainer for the model family."""
    if _is_tree(model):
        return shap.TreeExplainer(model)
    if hasattr(model, "coef_") and background is not None:
        return shap.LinearExplainer(model, background)
    # last resort — model-agnostic (slower); only hit by exotic estimators
    return shap.Explainer(model, background)


def _shap_2d(explainer, X_trans: np.ndarray) -> np.ndarray:
    """Return a clean (n_rows, n_features) SHAP matrix for the positive output.

    Handles the shape differences between binary RF (3-D: n×features×classes),
    binary XGB (2-D) and regression (2-D).
    """
    sv = explainer(X_trans, check_additivity=False)
    vals = np.asarray(sv.values)
    if vals.ndim == 3:           # (n, features, classes) -> take positive class
        vals = vals[:, :, -1]
    return vals


def _split_feature(name: str, categorical_cols: list[str]) -> tuple[str, str | None]:
    """Split a one-hot feature name 'col_value' into (col, value).

    Matches the longest categorical-column prefix so values containing
    underscores/spaces (e.g. 'Major Damage') survive intact.
    """
    for col in sorted(categorical_cols, key=len, reverse=True):
        if name == col:
            return col, None
        if name.startswith(col + "_"):
            return col, name[len(col) + 1:]
    return name, None  # numeric / engineered feature


def explain_instance(
    pipeline: Pipeline, raw_row: pd.DataFrame, stage: str,
    explainer=None, cfg: Config | None = None, top_n: int | None = None,
) -> list[dict]:
    """Return the top human-readable factors driving a single prediction.

    Args:
        pipeline: The fitted full pipeline (fraud or severity).
        raw_row: A 1-row DataFrame of RAW claim fields.
        stage: ``"fraud"`` or ``"severity"`` (controls the direction wording).
        explainer: Optional pre-built SHAP explainer (cached at serving for speed).
        cfg: Optional config.
        top_n: How many factors to return.

    Returns:
        A list of dicts: ``{reason, value, impact, direction}`` ready for the UI.

    """
    cfg = cfg or load_config()
    top_n = top_n or int(cfg.get("explainability.top_n_reasons"))
    categorical_cols = list(cfg.get("features.categorical"))

    preprocessor, model = unwrap(pipeline)
    X_trans = preprocessor.transform(raw_row)
    feat_names = get_feature_names(preprocessor)

    try:
        explainer = explainer or _build_explainer(model, X_trans)
        shap_vals = _shap_2d(explainer, X_trans)[0]
    except Exception as exc:  # noqa: BLE001 - explainability must never break a prediction
        log.warning("SHAP failed (%s); falling back to model importances", exc)
        shap_vals = _fallback_importance(model, X_trans[0], len(feat_names))

    # Only surface a categorical factor for the category the claim ACTUALLY has:
    # a one-hot column that is 0 for this row (an absent category) is not a real
    # driver of THIS prediction, even if SHAP assigns it a non-zero value. Numeric
    # features are always eligible.
    x_row = np.asarray(X_trans)[0]
    eligible: list[int] = []
    for idx in range(len(feat_names)):
        _, value = _split_feature(feat_names[idx], categorical_cols)
        is_categorical = value is not None
        if is_categorical and float(x_row[idx]) == 0.0:
            continue  # inactive one-hot category — skip
        eligible.append(idx)
    order = sorted(eligible, key=lambda i: abs(shap_vals[i]), reverse=True)[:top_n]

    increases = "increases fraud risk" if stage == "fraud" else "raises the estimate"
    decreases = "lowers fraud risk" if stage == "fraud" else "lowers the estimate"

    reasons: list[dict] = []
    for idx in order:
        name = feat_names[idx]
        col, value = _split_feature(name, categorical_cols)
        if value is not None:  # active one-hot categorical
            reason = f"{_pretty(col)} is '{value}'"
            shown_value = value
        else:                  # numeric / engineered
            raw_val = raw_row.iloc[0].get(col, "")
            reason = f"{_pretty(col)}"
            shown_value = raw_val
        reasons.append({
            "reason": reason,
            "value": _fmt(shown_value),
            "impact": round(float(shap_vals[idx]), 4),
            "direction": increases if shap_vals[idx] >= 0 else decreases,
        })
    return reasons


def _fmt(v) -> str:
    if isinstance(v, int | float | np.integer | np.floating) and not isinstance(v, bool):
        return f"{v:,.0f}" if abs(float(v)) >= 100 else f"{v}"
    return str(v)


def _fallback_importance(model, x_row: np.ndarray, n: int) -> np.ndarray:
    """Crude per-feature attribution if SHAP is unavailable for this model."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_[:n] * np.sign(x_row[:n])
    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)[:n]
        return coef * x_row[:n]
    return np.zeros(n)


def generate_shap_artifacts(
    pipeline: Pipeline, X_sample: pd.DataFrame, stage: str, cfg: Config | None = None
) -> None:
    """Save a global beeswarm + 2 local waterfall plots for the report.

    Args:
        pipeline: Fitted full pipeline.
        X_sample: Raw rows to explain (a sample of the test fold).
        stage: ``"fraud"`` or ``"severity"`` (used for filenames/titles).
        cfg: Optional config.

    """
    cfg = cfg or load_config()
    plots_dir = cfg.path("paths.plots_dir")
    bg = int(cfg.get("explainability.background_samples"))
    X_sample = X_sample.head(max(bg, 50)).reset_index(drop=True)

    preprocessor, model = unwrap(pipeline)
    X_trans = preprocessor.transform(X_sample)
    feat_names = get_feature_names(preprocessor)

    try:
        explainer = _build_explainer(model, X_trans)
        shap_matrix = _shap_2d(explainer, X_trans)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not generate SHAP artifacts for %s: %s", stage, exc)
        return

    # ── global beeswarm ──
    fig = plt.figure()
    shap.summary_plot(shap_matrix, features=X_trans, feature_names=feat_names,
                      show=False, max_display=15)
    plt.title(f"SHAP global feature importance — {stage} model")
    plt.tight_layout()
    fig.savefig(plots_dir / f"shap_summary_{stage}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── 2 local explanations (waterfall) ──
    base = explainer.expected_value
    base = float(np.ravel(base)[-1]) if np.ndim(base) else float(base)
    for i in range(min(2, len(X_sample))):
        expl = shap.Explanation(values=shap_matrix[i], base_values=base,
                                data=X_trans[i], feature_names=feat_names)
        fig = plt.figure()
        shap.plots.waterfall(expl, max_display=10, show=False)
        plt.title(f"SHAP local explanation #{i+1} — {stage}")
        plt.tight_layout()
        fig.savefig(plots_dir / f"shap_local_{stage}_{i+1}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    log.info("Saved SHAP artifacts for %s model", stage)
