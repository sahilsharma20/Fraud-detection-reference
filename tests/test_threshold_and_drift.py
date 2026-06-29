"""Unit tests for the cost matrix / threshold optimiser and the PSI drift maths."""

from __future__ import annotations

import numpy as np

from monitoring.drift_monitor import _psi
from src.threshold_optimizer import optimize_threshold, total_cost


def test_total_cost_counts_fn_and_fp():
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.9, 0.1, 0.2, 0.8])  # at 0.5 -> pred [1,0,0,1]
    cost, fn, fp = total_cost(y_true, y_proba, 0.5, cost_fn=30000, cost_fp=5000)
    assert fn == 1 and fp == 1            # missed the 3rd (0.2), flagged the 4th (0.8)
    assert cost == 30000 + 5000


def test_optimizer_beats_default_and_stays_in_bounds(cfg):
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.25, 500)
    # separable-ish scores so an interior optimum exists
    y_proba = np.clip(0.15 + 0.5 * y_true + rng.normal(0, 0.2, 500), 0, 1)
    res = optimize_threshold(y_true, y_proba, cfg, persist=False)  # don't clobber prod artifacts
    assert cfg.get("threshold.grid_start") <= res["chosen_threshold"] <= cfg.get("threshold.grid_stop")
    # the optimised threshold can never cost MORE than the naive 0.5 default
    assert res["chosen_cost"] <= res["default_0.5_cost"]


def test_psi_zero_for_identical_distribution():
    p = np.array([0.2, 0.3, 0.5])
    assert _psi(p, p) == 0.0 or _psi(p, p) < 1e-9


def test_psi_positive_for_shifted_distribution():
    expected = np.array([0.7, 0.2, 0.1])
    actual = np.array([0.1, 0.2, 0.7])     # mass moved across bins
    assert _psi(expected, actual) > 0.25   # clearly "significant" drift
