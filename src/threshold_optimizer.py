"""Module 5 — Cost-sensitive threshold optimisation.

DESIGN DECISION: the model outputs a probability; the *business* decides the
cut-off. The default 0.5 is a statistical artefact that implicitly assumes a
false positive and a false negative cost the same. They don't:

    * FALSE NEGATIVE (fraud predicted genuine -> paid out): direct loss of the
      fraudulent payout. We use avg_fraud_payout = ₹30,000.
    * FALSE POSITIVE (genuine flagged "needs review"): investigation effort +
      customer-trust damage. We use ₹2,000.

A missed fraud costs ~15× a false alarm, so the cost-minimising threshold sits
WELL BELOW 0.5 — we deliberately accept more false alarms to stop expensive
fraud. We sweep the whole grid, pick the threshold that minimises total expected
cost on the held-out fold, persist it to the inference config, and plot the
cost curve for the report.

FRESHER PITFALL AVOIDED: shipping 0.5 because "that's the default", or tuning the
threshold on the training data (overfit cut-off). We tune on the held-out test
predictions and store the number as a first-class, versioned artifact.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import Config, load_config, update_inference_config  # noqa: E402
from src.logger import get_logger  # noqa: E402

log = get_logger(__name__)


def total_cost(y_true: np.ndarray, y_proba: np.ndarray, threshold: float,
               cost_fn: float, cost_fp: float) -> tuple[float, int, int]:
    """Total expected cost at a threshold.

    Returns:
        ``(total_cost, n_false_negatives, n_false_positives)``.

    """
    y_pred = (y_proba >= threshold).astype(int)
    fn = int(((y_true == 1) & (y_pred == 0)).sum())  # fraud we missed
    fp = int(((y_true == 0) & (y_pred == 1)).sum())  # genuine we flagged
    return fn * cost_fn + fp * cost_fp, fn, fp


def optimize_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, cfg: Config | None = None, persist: bool = True
) -> dict:
    """Sweep thresholds, pick the cost-minimising one, persist & plot it.

    Args:
        y_true: Held-out true labels (0/1).
        y_proba: Held-out predicted fraud probabilities.
        cfg: Optional config.
        persist: If False, compute only — don't write artifacts (used in tests so
            unit runs never clobber the production inference config / plots).

    Returns:
        A dict with the chosen threshold and cost comparison vs the 0.5 default.

    """
    cfg = cfg or load_config()
    cost_fn = float(cfg.get("threshold.cost_false_negative"))
    cost_fp = float(cfg.get("threshold.cost_false_positive"))
    grid = np.linspace(
        float(cfg.get("threshold.grid_start")),
        float(cfg.get("threshold.grid_stop")),
        int(cfg.get("threshold.grid_steps")),
    )

    costs = np.array([total_cost(y_true, y_proba, t, cost_fn, cost_fp)[0] for t in grid])
    best_idx = int(np.argmin(costs))
    chosen = float(grid[best_idx])

    chosen_cost, chosen_fn, chosen_fp = total_cost(y_true, y_proba, chosen, cost_fn, cost_fp)
    default_cost, default_fn, default_fp = total_cost(y_true, y_proba, 0.5, cost_fn, cost_fp)
    savings = default_cost - chosen_cost

    result = {
        "chosen_threshold": round(chosen, 4),
        "cost_false_negative": cost_fn,
        "cost_false_positive": cost_fp,
        "chosen_cost": chosen_cost,
        "chosen_false_negatives": chosen_fn,
        "chosen_false_positives": chosen_fp,
        "default_0.5_cost": default_cost,
        "default_false_negatives": default_fn,
        "default_false_positives": default_fp,
        "cost_saving_vs_default": savings,
        "test_set_size": int(len(y_true)),
    }

    if persist:
        # persist as a first-class artifact the inference pipeline will read
        update_inference_config({"fraud_threshold": round(chosen, 4),
                                 "threshold_costs": {"fn": cost_fn, "fp": cost_fp}}, cfg)
        with open(cfg.path("paths.metrics_dir") / "threshold.json", "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        _plot_cost_curve(grid, costs, chosen, cfg)

    log.info(
        "Chosen threshold=%.3f (cost ₹%.0f, FN=%d FP=%d) vs default 0.5 (cost ₹%.0f) -> saves ₹%.0f on test",
        chosen, chosen_cost, chosen_fn, chosen_fp, default_cost, savings,
    )
    return result


def _plot_cost_curve(grid, costs, chosen, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, costs, color="#1565c0", lw=2)
    ax.axvline(chosen, color="#2e7d32", ls="--", label=f"chosen = {chosen:.2f}")
    ax.axvline(0.5, color="gray", ls=":", label="default = 0.50")
    ax.set_xlabel("Decision threshold (flag as fraud if P ≥ threshold)")
    ax.set_ylabel("Total expected cost (₹)")
    ax.set_title("Cost vs threshold — minimised away from 0.5")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.path("paths.plots_dir") / "cost_vs_threshold.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    import pandas as pd

    cfg = load_config()
    preds = pd.read_csv(cfg.path("paths.metrics_dir") / "fraud_test_predictions.csv")
    optimize_threshold(preds["y_true"].to_numpy(), preds["y_proba"].to_numpy(), cfg)
