"""Unit tests for the ranking-metric helpers.

These tests are pure-function checks — no DuckDB, no model fit, no
artefacts. Run in milliseconds.
"""

from __future__ import annotations

import math

import numpy as np

from shortage_forecast.baseline import compute_metrics, precision_at_k
from shortage_forecast.operational import (
    _within_month_rank,
    score_blended,
    score_heuristic_compound,
    score_heuristic_single,
)

# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

def test_precision_at_k_perfect_ranking():
    """Top-k matches all positives -> precision = 1.0."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.9, 0.8])  # top 2 are the positives
    assert precision_at_k(y_true, y_score, k=2) == 1.0


def test_precision_at_k_no_positives_in_topk():
    y_true = np.array([1, 1, 0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.5, 0.7, 0.9])  # top 2 are the negatives
    assert precision_at_k(y_true, y_score, k=2) == 0.0


def test_precision_at_k_partial_recall():
    y_true = np.array([1, 0, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.4, 0.3, 0.2])
    # Top 2 = indices [0, 1] → one positive, one negative.
    assert precision_at_k(y_true, y_score, k=2) == 0.5


def test_precision_at_k_returns_nan_when_k_exceeds_n():
    """Stratified slices smaller than K should not crash; they return NaN."""
    y_true = np.array([1, 0])
    y_score = np.array([0.5, 0.5])
    assert math.isnan(precision_at_k(y_true, y_score, k=10))


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def test_compute_metrics_handles_single_class_input():
    """When y_true has only one class, ranking metrics are NaN, not raises."""
    y_true = np.zeros(10, dtype=int)
    y_score = np.linspace(0, 1, 10)
    out = compute_metrics(y_true, y_score, top_k_values=[5])
    assert math.isnan(out["roc_auc"])
    assert math.isnan(out["pr_auc"])
    # Brier is still defined when y_true is all zeros.
    assert not math.isnan(out["brier"])
    assert math.isnan(out["precision_at_5"])


def test_compute_metrics_balanced_input_returns_finite():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_score = rng.random(200)
    out = compute_metrics(y_true, y_score, top_k_values=[10, 50])
    for key in ("roc_auc", "pr_auc", "brier", "precision_at_10", "precision_at_50"):
        assert not math.isnan(out[key]), f"{key} should be finite"


# ---------------------------------------------------------------------------
# _within_month_rank
# ---------------------------------------------------------------------------

def test_within_month_rank_normalises_per_month():
    """Each month's ranks span the same range regardless of raw scale.

    This is the property `score_blended` relies on.
    """
    scores = np.array([0.1, 0.5, 0.9,        # month A — small
                       100.0, 200.0, 300.0]) # month B — much larger
    months = np.array(["A", "A", "A", "B", "B", "B"])
    ranks = _within_month_rank(scores, months)
    assert list(ranks[:3]) == [1.0, 2.0, 3.0]
    assert list(ranks[3:]) == [1.0, 2.0, 3.0]


def test_within_month_rank_handles_ties_with_average():
    scores = np.array([1.0, 1.0, 2.0])
    months = np.array(["A", "A", "A"])
    ranks = _within_month_rank(scores, months)
    # Tied at rank 1+2 -> 1.5 each; the unique value gets rank 3.
    assert list(ranks) == [1.5, 1.5, 3.0]


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

def test_score_heuristic_single_orders_by_shortages_prior_12m():
    import pandas as pd
    df = pd.DataFrame({"shortages_prior_12m": [0, 5, 1, 3]})
    scores = score_heuristic_single(df)
    # Argmax must be the row with 5 prior shortages.
    assert int(scores.argmax()) == 1


def test_score_heuristic_compound_uses_mfr_rate_as_tiebreaker():
    import pandas as pd
    df = pd.DataFrame({
        "shortages_prior_12m":   [2, 2, 2, 1],
        "mfr_shortage_rate_12m": [0.1, 0.5, 0.3, 0.9],
    })
    scores = score_heuristic_compound(df)
    # Among the three tied at primary=2, the one with mfr_rate=0.5 wins.
    top = int(scores.argmax())
    assert top == 1


# ---------------------------------------------------------------------------
# score_blended
# ---------------------------------------------------------------------------

def test_score_blended_respects_within_month_partition():
    """Blending should never let a high score in month A out-rank an item in
    month B — the ranks reset per month before being summed.
    """
    import pandas as pd
    test = pd.DataFrame({
        "observation_date": pd.to_datetime(
            ["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"]
        ),
    })
    # GBM scores: month A has tiny scores; month B has huge ones.
    gbm = np.array([0.01, 0.02, 99.0, 100.0])
    heur = np.array([5.0, 1.0, 1.0, 5.0])

    blended = score_blended(test, gbm, heur, gbm_weight=2.0, heuristic_weight=1.0)

    # In month A (idx 0,1): GBM ranks [1,2], heur ranks [2,1] -> blended [4,5]
    # In month B (idx 2,3): GBM ranks [1,2], heur ranks [1,2] -> blended [3,6]
    assert blended[0] == 2 * 1 + 1 * 2
    assert blended[1] == 2 * 2 + 1 * 1
    assert blended[2] == 2 * 1 + 1 * 1
    assert blended[3] == 2 * 2 + 1 * 2
