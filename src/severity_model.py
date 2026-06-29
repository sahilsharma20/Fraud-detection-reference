"""Modules 6 & 7 — Stage 2: Severity regression, with TWO leakage guards.

THE HEADLINE LEAKAGE DECISION (Module 6)
----------------------------------------
The severity model is trained on **genuine claims ONLY** — every row where
``fraud_reported == 'Y'`` is dropped before training. Rationale:

    A fraudulent claim's ``total_claim_amount`` is FABRICATED — it reflects what
    the fraudster tried to extract, not the true cost of a real incident. If we
    train the reserving model on those amounts, we teach it that (say) a minor
    fender-bender is worth ₹80,000 because a fraudster inflated it. That poisons
    every future genuine prediction. In production we ONLY ever predict severity
    for claims the Stage-1 model judged genuine, so we must also TRAIN only on
    genuine claims — train/serve populations must match.

THE SECOND LEAKAGE GUARD (target-composition, Module 3/7)
---------------------------------------------------------
``total_claim_amount == injury_claim + property_claim + vehicle_claim`` exactly.
Those three component columns are excluded from the severity feature set (see
``build_preprocessor('severity')`` / config.severity_leakage_columns) — otherwise
the model trivially reconstructs the target (R²≈1.0) and learns nothing useful.

OTHER DECISIONS
---------------
* Right-skewed target -> we fit on ``log1p(y)`` via ``TransformedTargetRegressor``
  and auto-invert with ``expm1`` at predict time, so callers always get rupees,
  never log-space. Metrics are computed in rupee space too.
* Errors are translated into a business sentence (avg reserving error per claim),
  because "RMSE=11240" means nothing to a claims manager.
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from src.config import Config, load_config, update_inference_config  # noqa: E402
from src.feature_engineering import build_preprocessor  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.tracking import setup_mlflow  # noqa: E402

log = get_logger(__name__)


def _build_regressor(name: str, cfg: Config):
    seed = int(cfg.get("project.random_state"))
    if name == "linear_regression":
        return LinearRegression()
    if name == "random_forest":
        p = cfg.get("severity_model.random_forest")
        return RandomForestRegressor(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"], random_state=seed, n_jobs=-1,
        )
    if name == "xgboost":
        p = cfg.get("severity_model.xgboost")
        return XGBRegressor(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            learning_rate=p["learning_rate"], subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"], random_state=seed,
            n_jobs=-1, tree_method="hist",
        )
    raise ValueError(f"Unknown severity model '{name}'")


def _wrap_target(regressor, cfg: Config):
    """Wrap a regressor so it trains on log1p(y) and predicts in rupee space."""
    if bool(cfg.get("severity_model.log_transform_target")):
        return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)
    return regressor


def _genuine_only(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """LEAKAGE GUARD: keep only non-fraud claims for severity training/eval."""
    fraud_col = cfg.get("data.target_fraud")
    pos = cfg.get("data.fraud_positive_label")
    genuine = df[df[fraud_col] != pos].copy()
    log.info("Severity data: %d/%d rows kept (dropped %d fraud-flagged claims)",
             len(genuine), len(df), len(df) - len(genuine))
    return genuine


def _split_xy(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, np.ndarray]:
    sev_col = cfg.get("data.target_severity")
    fraud_col = cfg.get("data.target_fraud")
    # drop the target AND the fraud label from X (label isn't a feature here)
    X = df.drop(columns=[sev_col, fraud_col])
    y = df[sev_col].to_numpy(dtype=float)
    return X, y


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2_score(y_true, y_pred)),
        "mean_bias": float(np.mean(y_pred - y_true)),  # +ve => over-reserves on average
    }


def train_severity_models(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Config | None = None
) -> dict:
    """Train severity candidates on GENUINE claims only; register & persist best.

    Returns:
        Results dict (per-model metrics, best model, business statement).

    """
    cfg = cfg or load_config()
    setup_mlflow(cfg, cfg.get("mlflow.experiment_severity"))

    # ── leakage guard: genuine claims only, for both train and evaluation ──
    train_g = _genuine_only(train_df, cfg)
    test_g = _genuine_only(test_df, cfg)
    X_train, y_train = _split_xy(train_g, cfg)
    X_test, y_test = _split_xy(test_g, cfg)
    primary = cfg.get("severity_model.primary_metric")  # rmse -> lower is better

    results: dict[str, dict] = {}
    best = {"name": None, "score": np.inf, "pipeline": None, "pred": None, "run_id": None}

    for name in cfg.get("severity_model.candidates"):
        with mlflow.start_run(run_name=f"severity_{name}") as run:
            reg = _wrap_target(_build_regressor(name, cfg), cfg)
            pipe = Pipeline([("preprocess", build_preprocessor("severity", cfg)), ("model", reg)])
            pipe.fit(X_train, y_train)

            pred = pipe.predict(X_test)
            m = _metrics(y_test, pred)
            results[name] = m

            mlflow.log_param("model_type", name)
            mlflow.log_param("log_transform_target", bool(cfg.get("severity_model.log_transform_target")))
            mlflow.log_metrics(m)
            mlflow.sklearn.log_model(pipe, artifact_path="model")

            log.info("[%s] RMSE=%.0f MAE=%.0f R2=%.3f bias=%.0f",
                     name, m["rmse"], m["mae"], m["r2"], m["mean_bias"])
            if m[primary] < best["score"]:
                best.update(name=name, score=m[primary], pipeline=pipe,
                            pred=pred, run_id=run.info.run_id)

    # register winner in the MLflow Model Registry
    registry_name = cfg.get("mlflow.registered_severity_model")
    mv = mlflow.register_model(f"runs:/{best['run_id']}/model", registry_name)
    log.info("Registered '%s' v%s (best=%s, RMSE=%.0f)",
             registry_name, mv.version, best["name"], best["score"])

    # persist best pipeline + write MAE into the inference config (used for the
    # ± error band shown to users, and the reserving-impact figure in the report)
    joblib.dump(best["pipeline"], cfg.path("paths.severity_model"))
    best_mae = results[best["name"]]["mae"]
    update_inference_config({"severity_mae": round(best_mae, 2),
                             "severity_model": best["name"]}, cfg)

    currency = cfg.get("business.currency_symbol")
    business_statement = (
        f"On average the reserving model's estimate is off by {currency}{best_mae:,.0f} "
        f"per genuine claim (MAE). Mean bias {results[best['name']]['mean_bias']:+,.0f} "
        f"means it tends to {'over' if results[best['name']]['mean_bias'] >= 0 else 'under'}-reserve."
    )
    log.info(business_statement)

    _save_plots(y_test, best["pred"], best["name"], cfg)
    payload = {"per_model": results, "best_model": best["name"],
               "business_statement": business_statement, "n_test_genuine": int(len(y_test))}
    with open(cfg.path("paths.metrics_dir") / "severity_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    return {"results": results, "best_model": best["name"],
            "registry_version": mv.version, "business_statement": business_statement}


def _save_plots(y_true, y_pred, model_name, cfg: Config) -> None:
    residuals = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(y_true, y_pred, alpha=0.4, s=14, color="#1565c0")
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1)
    axes[0].set_xlabel("actual total_claim_amount")
    axes[0].set_ylabel("predicted")
    axes[0].set_title(f"Predicted vs actual — {model_name}")

    axes[1].hist(residuals, bins=40, color="#6a1b9a", alpha=0.8)
    axes[1].axvline(0, color="red", ls="--")
    axes[1].set_xlabel("prediction error (predicted − actual)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Severity error distribution")
    fig.tight_layout()
    fig.savefig(cfg.path("paths.plots_dir") / "severity_errors.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    from src.data_ingestion import ingest

    tr, te = ingest()
    train_severity_models(tr, te)
