"""Module 4 — Stage 1: Fraud classification (LR -> RF -> XGBoost) with MLflow.

DESIGN DECISIONS
----------------
* Three models of increasing capacity (logistic regression baseline -> random
  forest -> gradient boosting). The baseline is not ceremony: if XGBoost can't
  beat a regularised linear model, that's a signal the features (not the model)
  are the problem — a senior always keeps an honest baseline.
* Every candidate is the FULL pipeline (preprocessor + estimator), fit only on
  train, evaluated on the held-out test fold, and tracked in MLflow (params,
  metrics, artifacts). The single best model (by PR-AUC) is REGISTERED in the
  MLflow Model Registry — not merely logged — so there is a named, versioned,
  promotable artifact, which is what production teams actually deploy from.

WHY PR-AUC, NOT ACCURACY OR ROC-AUC (the core fresher trap)
-----------------------------------------------------------
* Accuracy: with ~22% fraud, "always predict genuine" scores ~78% and catches
  zero fraud. Accuracy rewards the majority class — useless for a rare-event
  hunt.
* ROC-AUC: better, but ROC uses the true-negative rate, and on imbalanced data
  the huge negative class makes ROC-AUC look flatteringly high even when the
  model is poor at the thing we care about (finding the few positives).
* PR-AUC (average precision): summarises the precision/recall trade-off over the
  POSITIVE (fraud) class only. It moves when the model gets better at fraud
  specifically, so it is the right model-selection metric here. We still LOG
  accuracy + ROC-AUC so the report can show *why* they mislead.

Class imbalance is handled by the cost-sensitive THRESHOLD (Module 5), not by
class reweighting — at ~27% prevalence (moderate), reweighting would inflate the
predicted probabilities and corrupt the cost optimisation that depends on them.
We pick one lever and keep probabilities calibrated. (Reweighting is a config
toggle — ``fraud_model.handle_imbalance_via_weights`` — for rarer-event regimes.)
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
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src.config import Config, load_config  # noqa: E402
from src.feature_engineering import build_preprocessor  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.tracking import setup_mlflow  # noqa: E402

log = get_logger(__name__)


def _build_classifier(name: str, cfg: Config, n_pos: int, n_neg: int):
    """Instantiate a classifier by config name, wiring in imbalance handling."""
    seed = int(cfg.get("project.random_state"))
    if name == "logistic_regression":
        p = cfg.get("fraud_model.logistic_regression")
        return LogisticRegression(
            C=p["C"], max_iter=p["max_iter"], class_weight=p["class_weight"], random_state=seed
        )
    if name == "random_forest":
        p = cfg.get("fraud_model.random_forest")
        return RandomForestClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"], class_weight=p["class_weight"],
            random_state=seed, n_jobs=-1,
        )
    if name == "xgboost":
        p = cfg.get("fraud_model.xgboost")
        # scale_pos_weight = negatives/positives is XGBoost's imbalance knob, but we
        # keep it at 1 (off) on purpose — imbalance is handled by the cost threshold
        # (see config.fraud_model.handle_imbalance_via_weights). Toggling that flag
        # to true re-enables reweighting without code changes.
        spw = (n_neg / max(n_pos, 1)) if cfg.get("fraud_model.handle_imbalance_via_weights") else 1.0
        return XGBClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            learning_rate=p["learning_rate"], subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"], scale_pos_weight=spw,
            eval_metric="aucpr", random_state=seed, n_jobs=-1, tree_method="hist",
        )
    raise ValueError(f"Unknown fraud model '{name}'")


def _evaluate(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute the full metric suite at a given threshold (default 0.5)."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),  # logged to SHOW it misleads
    }


def _split_xy(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, np.ndarray]:
    fraud_col = cfg.get("data.target_fraud")
    pos = cfg.get("data.fraud_positive_label")
    X = df.drop(columns=[fraud_col])
    y = (df[fraud_col] == pos).astype(int).to_numpy()
    return X, y


def train_fraud_models(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Config | None = None
) -> dict:
    """Train all fraud candidates, track in MLflow, register & persist the best.

    Args:
        train_df: Training fold (raw columns; the pipeline preprocesses).
        test_df: Held-out test fold.
        cfg: Optional config.

    Returns:
        A results dict with per-model metrics, the chosen model name, and the
        held-out ``(y_true, y_proba)`` arrays (consumed by the threshold optimizer).

    """
    cfg = cfg or load_config()
    setup_mlflow(cfg, cfg.get("mlflow.experiment_fraud"))

    X_train, y_train = _split_xy(train_df, cfg)
    X_test, y_test = _split_xy(test_df, cfg)
    n_pos, n_neg = int(y_train.sum()), int((1 - y_train).sum())
    primary = cfg.get("fraud_model.primary_metric")

    results: dict[str, dict] = {}
    best = {"name": None, "score": -np.inf, "pipeline": None, "proba": None, "run_id": None}

    for name in cfg.get("fraud_model.candidates"):
        with mlflow.start_run(run_name=f"fraud_{name}") as run:
            clf = _build_classifier(name, cfg, n_pos, n_neg)
            pipe = Pipeline([("preprocess", build_preprocessor("fraud", cfg)), ("model", clf)])
            pipe.fit(X_train, y_train)

            proba = pipe.predict_proba(X_test)[:, 1]
            metrics = _evaluate(y_test, proba)
            results[name] = metrics

            mlflow.log_param("model_type", name)
            mlflow.log_param("reweight_imbalance", bool(cfg.get("fraud_model.handle_imbalance_via_weights")))
            mlflow.log_param("train_pos_rate", round(n_pos / max(n_pos + n_neg, 1), 3))
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, artifact_path="model")

            log.info(
                "[%s] PR-AUC=%.3f ROC-AUC=%.3f P=%.3f R=%.3f F1=%.3f (acc=%.3f <- misleading)",
                name, metrics["pr_auc"], metrics["roc_auc"], metrics["precision"],
                metrics["recall"], metrics["f1"], metrics["accuracy"],
            )
            if metrics[primary] > best["score"]:
                best.update(name=name, score=metrics[primary], pipeline=pipe,
                            proba=proba, run_id=run.info.run_id)

    # ── register the winner in the MLflow Model Registry (versioned, promotable) ──
    registry_name = cfg.get("mlflow.registered_fraud_model")
    model_uri = f"runs:/{best['run_id']}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=registry_name)
    log.info("Registered '%s' v%s in MLflow Model Registry (best=%s, %s=%.3f)",
             registry_name, mv.version, best["name"], primary, best["score"])

    # ── persist the best full pipeline for low-dependency serving (no MLflow at runtime) ──
    cfg.path("paths.models_dir").mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], cfg.path("paths.fraud_model"))
    log.info("Saved best fraud pipeline -> %s", cfg.path("paths.fraud_model"))

    _save_pr_curve(y_test, best["proba"], best["name"], cfg)
    _save_metrics(results, best["name"], y_test, best["proba"], cfg)

    return {
        "results": results,
        "best_model": best["name"],
        "registry_version": mv.version,
        "y_test": y_test,
        "proba_test": best["proba"],
    }


def _save_pr_curve(y_true, proba, model_name, cfg: Config) -> None:
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    baseline = float(np.mean(y_true))  # a random classifier's precision = prevalence
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#c62828", lw=2, label=f"{model_name} (AP={ap:.3f})")
    ax.axhline(baseline, ls="--", color="gray", label=f"random baseline ({baseline:.2f})")
    ax.set_xlabel("Recall (fraud caught)")
    ax.set_ylabel("Precision (flags that are real fraud)")
    ax.set_title("Precision–Recall curve — Stage 1 fraud model")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(cfg.path("paths.plots_dir") / "pr_curve.png", dpi=120)
    plt.close(fig)


def _save_metrics(results, best_name, y_true, proba, cfg: Config) -> None:
    cm = confusion_matrix(y_true, (proba >= 0.5).astype(int)).tolist()
    payload = {"per_model": results, "best_model": best_name,
               "confusion_matrix_at_0.5": cm, "test_fraud_rate": float(np.mean(y_true))}
    out = cfg.path("paths.metrics_dir") / "fraud_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    # also persist held-out predictions so the threshold optimizer can run standalone
    pd.DataFrame({"y_true": y_true, "y_proba": proba}).to_csv(
        cfg.path("paths.metrics_dir") / "fraud_test_predictions.csv", index=False
    )


if __name__ == "__main__":
    from src.data_ingestion import ingest

    tr, te = ingest()
    train_fraud_models(tr, te)
