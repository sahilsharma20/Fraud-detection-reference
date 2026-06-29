"""Module 2 — Exploratory Data Analysis (artifact-generating, not notebook-style).

This runs as part of the training pipeline and writes plots + a JSON summary that
the report (Deliverable C) embeds. EDA here is *reproducible code*, not a throwaway
notebook — anyone can regenerate every figure with `make train`.

KEY ANALYTICAL POINT (and the headline fresher mistake):
    The fraud target is imbalanced (~25% positive in this dataset, far less in
    the real world). **Plain accuracy is misleading**: a model that predicts
    "never fraud" scores ~75% accuracy here (and ~99% on a realistic 1% base
    rate) while catching ZERO fraud — useless. That is exactly why downstream we
    optimise Precision / Recall / F1 / PR-AUC and a cost-weighted threshold, not
    accuracy. The class-balance plot makes this visceral for the report reader.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # headless backend — never tries to open a window (CI/Docker safe)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import Config, load_config  # noqa: E402
from src.logger import get_logger  # noqa: E402

log = get_logger(__name__)
sns.set_theme(style="whitegrid")


def _savefig(fig: plt.Figure, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot -> %s", path)


def run_eda(train_df: pd.DataFrame, cfg: Config | None = None) -> dict:
    """Compute EDA summary stats and save report-ready plots.

    Args:
        train_df: The TRAINING fold only (EDA on test would be peeking).
        cfg: Optional config.

    Returns:
        A JSON-serialisable summary dict (also written to metrics/eda_summary.json).

    """
    cfg = cfg or load_config()
    plots_dir = cfg.path("paths.plots_dir")
    metrics_dir = cfg.path("paths.metrics_dir")
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    fraud_col = cfg.get("data.target_fraud")
    sev_col = cfg.get("data.target_severity")

    fraud_rate = float((train_df[fraud_col] == "Y").mean())
    target_skew = float(train_df[sev_col].skew())

    # ── 1. Class balance (why accuracy lies) ──
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = train_df[fraud_col].value_counts()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                palette=["#2e7d32", "#c62828"], legend=False, ax=ax)
    ax.set_title(f"Fraud class balance (fraud = {fraud_rate*100:.1f}%)")
    ax.set_xlabel("fraud_reported")
    ax.set_ylabel("claims")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    _savefig(fig, plots_dir / "class_balance.png")

    # ── 2. Target distribution: raw (skewed) vs log1p (why we log-transform) ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(train_df[sev_col], bins=40, kde=True, color="#1565c0", ax=axes[0])
    axes[0].set_title(f"total_claim_amount (skew={target_skew:.2f})")
    sns.histplot(np.log1p(train_df[sev_col]), bins=40, kde=True, color="#6a1b9a", ax=axes[1])
    axes[1].set_title("log1p(total_claim_amount) — near-normal")
    _savefig(fig, plots_dir / "target_distribution.png")

    # ── 3. Missing values per column ──
    # count both real NaN and the in-band '?' marker used by the raw file
    missing = (train_df.isna() | (train_df == "?")).sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing):
        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(missing))))
        sns.barplot(x=missing.values, y=missing.index, hue=missing.index,
                    palette="rocket", legend=False, ax=ax)
        ax.set_title("Missing / unknown values by column")
        ax.set_xlabel("count")
        _savefig(fig, plots_dir / "missing_values.png")

    # ── 4. Numeric correlation with fraud (signal sniff) ──
    num_cols = train_df.select_dtypes(include="number").columns
    fraud_bin = (train_df[fraud_col] == "Y").astype(int)
    corr = train_df[num_cols].corrwith(fraud_bin).abs().sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=corr.values, y=corr.index, hue=corr.index, palette="mako", legend=False, ax=ax)
    ax.set_title("|correlation| of numeric features with fraud")
    ax.set_xlabel("|Pearson r|")
    _savefig(fig, plots_dir / "fraud_correlation.png")

    # cardinality of categoricals (informs one-hot blow-up risk)
    cat_cols = train_df.select_dtypes(include="object").columns
    cardinality = {c: int(train_df[c].nunique()) for c in cat_cols}

    summary = {
        "n_rows_train": int(len(train_df)),
        "n_features": int(train_df.shape[1]),
        "fraud_rate": round(fraud_rate, 4),
        "majority_class_accuracy": round(1 - fraud_rate, 4),  # the "predict genuine" trap
        "target_skew_raw": round(target_skew, 3),
        "target_skew_log1p": round(float(np.log1p(train_df[sev_col]).skew()), 3),
        "missing_by_column": {k: int(v) for k, v in missing.items()},
        "categorical_cardinality": cardinality,
    }
    with open(metrics_dir / "eda_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log.info(
        "EDA done. fraud_rate=%.1f%%  majority-class-accuracy trap=%.1f%%  target skew %.2f -> %.2f (log)",
        fraud_rate * 100, summary["majority_class_accuracy"] * 100,
        summary["target_skew_raw"], summary["target_skew_log1p"],
    )
    return summary


if __name__ == "__main__":
    from src.data_ingestion import ingest

    train_df, _ = ingest()
    run_eda(train_df)
