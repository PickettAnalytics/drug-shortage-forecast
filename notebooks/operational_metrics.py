"""
Per-month operational metrics for the drug shortage model, with heuristic
baselines for honest portfolio comparison.

Extends the per-month Precision@K analysis to compare four rankers:

  1. LightGBM — the full feature-set model from baseline.py
  2. Logistic — the simple linear baseline from baseline.py
  3. Heuristic_1 — rank by shortages_prior_12m alone (crude "how many
     shortages has this drug had recently?" rule of thumb)
  4. Heuristic_2 — rank by shortages_prior_12m + mfr_shortage_rate_12m
     as tiebreaker (what a domain expert might build without ML)

The key question this answers: how much of the model's value comes from
the ML, vs being available via a sensible hand-built ranking?

Models are imported from baseline.py so the comparison always reflects the
current baseline configuration. Both heuristics use deterministic
tiebreaking so results are reproducible.

Usage:
    python operational_metrics.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import load_splits, TARGET
from baseline import fit_lightgbm, fit_logistic, predict_proba


TOP_K_OPERATIONAL = [10, 25, 50, 100]
RANDOM_TIEBREAK_SEED = 42


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    n = len(y_true)
    if n < k:
        return float("nan")
    top_idx = np.argpartition(-y_score, k - 1)[:k]
    return float(y_true[top_idx].sum()) / k


def score_heuristic_single(test: pd.DataFrame) -> np.ndarray:
    """Crude ranker: sort by shortages_prior_12m, random tiebreak.

    Many drugs have 0 or 1 prior shortages, so tiebreaking matters a lot.
    We add tiny deterministic noise so the heuristic can at least separate
    drugs with identical integer counts.
    """
    primary = test["shortages_prior_12m"].to_numpy(dtype=float)
    rng = np.random.RandomState(RANDOM_TIEBREAK_SEED)
    jitter = rng.rand(len(test)) * 1e-6
    return primary + jitter


def score_heuristic_compound(test: pd.DataFrame) -> np.ndarray:
    """Compound ranker: shortages_prior_12m primary, mfr_shortage_rate_12m secondary.

    Represents what a thoughtful analyst might build by hand: 'prioritize
    drugs with recent shortages; among those tied, prefer ones from
    struggling manufacturers.'
    """
    primary   = test["shortages_prior_12m"].to_numpy(dtype=float)
    secondary = test["mfr_shortage_rate_12m"].to_numpy(dtype=float)
    rng = np.random.RandomState(RANDOM_TIEBREAK_SEED)
    jitter = rng.rand(len(test)) * 1e-9
    # primary dominates; secondary acts as tiebreaker; jitter breaks remaining ties
    return primary + 0.01 * secondary + jitter


def _within_month_rank(scores: np.ndarray, observation_dates: np.ndarray) -> np.ndarray:
    """Per-month rank of each row (higher score = higher rank). Ties are
    broken by the row's average rank to keep the transform stable.

    Within-month ranks put every month on the same scale, which is what
    we need to combine signals that have different per-month calibration
    (the GBM under-scores quiet months; the heuristic doesn't)."""
    out = np.empty(len(scores), dtype=float)
    series = pd.Series(scores)
    for date in pd.unique(observation_dates):
        mask = observation_dates == date
        out[mask] = series[mask].rank(method="average").to_numpy()
    return out


def score_blended(
    test: pd.DataFrame,
    gbm_scores: np.ndarray,
    heuristic_scores: np.ndarray,
    gbm_weight: float = 2.0,
    heuristic_weight: float = 1.0,
) -> np.ndarray:
    """Weighted sum of within-month ranks of the GBM and heuristic scores.

    The GBM is well-calibrated across months and dominant past K=10; the
    heuristic is robust at the very top but blind to everything else.
    Adding within-month ranks puts both on the same scale; weighting
    GBM 2:1 lets the heuristic break top-of-rank ties without overriding
    the GBM at deeper K."""
    obs_dates = test["observation_date"].to_numpy()
    return (
        gbm_weight * _within_month_rank(gbm_scores, obs_dates)
        + heuristic_weight * _within_month_rank(heuristic_scores, obs_dates)
    )


def per_month_metrics(
    test: pd.DataFrame,
    scores: np.ndarray,
    name: str,
) -> pd.DataFrame:
    """Compute per-month precision@K for a given ranker."""
    rows = []
    for obs_date, group in test.groupby("observation_date", observed=True):
        idx = group.index.to_numpy()
        # Positional indices within the test frame, since test has a reset index
        pos_idx = np.where(test["observation_date"].to_numpy() == obs_date)[0]
        scores_month = scores[pos_idx]
        y_month = test.loc[idx, TARGET].to_numpy()
        row = {
            "ranker": name,
            "observation_date": obs_date,
            "n_drugs": len(group),
            "n_positives": int(y_month.sum()),
        }
        for k in TOP_K_OPERATIONAL:
            pk = precision_at_k(y_month, scores_month, k)
            row[f"precision_at_{k}"] = pk
            row[f"hits_at_{k}"] = pk * k if not np.isnan(pk) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("observation_date")


def summarize(monthly: pd.DataFrame, mean_base: float) -> pd.DataFrame:
    """Summary of per-month Precision@K across test months."""
    summary_rows = []
    for k in TOP_K_OPERATIONAL:
        col = f"precision_at_{k}"
        pk = monthly[col]
        summary_rows.append({
            "k":              k,
            "mean_precision": pk.mean(),
            "median":         pk.median(),
            "min":            pk.min(),
            "max":            pk.max(),
            "mean_hits":      pk.mean() * k,
            "lift_vs_random": pk.mean() / mean_base if mean_base > 0 else float("nan"),
        })
    return pd.DataFrame(summary_rows)


def print_summary_table(summary: pd.DataFrame, label: str) -> None:
    print(f"\n--- {label} ---")
    print(f"{'K':<6} {'mean':>7} {'median':>7} {'min':>7} {'max':>7} "
          f"{'mean hits':>11} {'lift':>7}")
    for _, row in summary.iterrows():
        print(f"{int(row['k']):<6} "
              f"{row['mean_precision']:>7.3f} "
              f"{row['median']:>7.3f} "
              f"{row['min']:>7.3f} "
              f"{row['max']:>7.3f} "
              f"{row['mean_hits']:>7.1f}/{int(row['k']):<3} "
              f"{row['lift_vs_random']:>6.1f}x")


def main() -> None:
    # --- Load & fit ---
    train, val, test = load_splits(verbose=False)
    print(f"Test set: {len(test):,} rows across "
          f"{test['observation_date'].nunique()} months")
    print(f"Test positive rate: {test[TARGET].mean():.4f}")

    print("\nFitting LightGBM (from baseline.py)...")
    gbm = fit_lightgbm(train, val)
    print(f"LightGBM best iteration: {gbm.best_iteration_}")
    print("Fitting logistic regression (from baseline.py)...")
    lr = fit_logistic(train, val)

    # --- Score rankers on test ---
    gbm_scores    = predict_proba(gbm, test)
    lr_scores     = predict_proba(lr,  test)
    heur_1_scores = score_heuristic_single(test)
    heur_2_scores = score_heuristic_compound(test)
    blend_scores  = score_blended(test, gbm_scores, heur_1_scores)

    monthly_gbm   = per_month_metrics(test, gbm_scores,    "lightgbm")
    monthly_lr    = per_month_metrics(test, lr_scores,     "logistic")
    monthly_h1    = per_month_metrics(test, heur_1_scores, "heuristic_single")
    monthly_h2    = per_month_metrics(test, heur_2_scores, "heuristic_compound")
    monthly_blend = per_month_metrics(test, blend_scores,  "blend_gbm_heur")

    mean_base = monthly_gbm["n_positives"].sum() / monthly_gbm["n_drugs"].sum()

    # --- Print per-month hits@10 for the four rankers ---
    print("\n" + "=" * 100)
    print("PER-MONTH hits@10 ACROSS RANKERS")
    print("=" * 100)
    merged = monthly_gbm[["observation_date", "n_positives", "hits_at_10"]].rename(
        columns={"hits_at_10": "gbm"})
    for name, df in [
        ("logistic", monthly_lr),
        ("heur_1",   monthly_h1),
        ("heur_2",   monthly_h2),
        ("blend",    monthly_blend),
    ]:
        merged = pd.merge(
            merged,
            df[["observation_date", "hits_at_10"]].rename(columns={"hits_at_10": name}),
            on="observation_date",
        )
    for col in ["gbm", "logistic", "heur_1", "heur_2", "blend"]:
        merged[col] = merged[col].astype(int).astype(str) + "/10"
    print(merged.to_string(index=False))

    # --- Summaries ---
    print("\n" + "=" * 100)
    print(f"SUMMARY — per-month Precision@K (mean base rate = {mean_base:.4f})")
    print("=" * 100)

    summary_gbm   = summarize(monthly_gbm,   mean_base)
    summary_lr    = summarize(monthly_lr,    mean_base)
    summary_h1    = summarize(monthly_h1,    mean_base)
    summary_h2    = summarize(monthly_h2,    mean_base)
    summary_blend = summarize(monthly_blend, mean_base)

    print_summary_table(summary_gbm,   "LightGBM (monotone-constrained)")
    print_summary_table(summary_lr,    "Logistic regression (baseline.py)")
    print_summary_table(summary_h1,    "Heuristic 1: shortages_prior_12m only")
    print_summary_table(summary_h2,    "Heuristic 2: shortages_prior_12m + mfr_rate tiebreak")
    print_summary_table(summary_blend, "Blend: within-month rank(GBM) + rank(heur_1)")

    # --- Headline comparisons ---
    print("\n" + "=" * 100)
    print("HEADLINE — Blend vs GBM vs heuristics (per-month mean precision)")
    print("=" * 100)
    for k in TOP_K_OPERATIONAL:
        blend_mean = summary_blend.loc[summary_blend["k"] == k, "mean_precision"].iloc[0]
        gbm_mean   = summary_gbm.loc[summary_gbm["k"]    == k, "mean_precision"].iloc[0]
        lr_mean    = summary_lr.loc[summary_lr["k"]     == k, "mean_precision"].iloc[0]
        h1_mean    = summary_h1.loc[summary_h1["k"]     == k, "mean_precision"].iloc[0]
        h2_mean    = summary_h2.loc[summary_h2["k"]     == k, "mean_precision"].iloc[0]

        def lift(target: float) -> str:
            if target <= 0:
                return "  n/a"
            return f"{(blend_mean - target) / target:+.1%}"

        print(f"  K={k:<4}  Blend {blend_mean:.3f}   "
              f"vs GBM {gbm_mean:.3f} ({lift(gbm_mean)})   "
              f"vs LR {lr_mean:.3f} ({lift(lr_mean)})   "
              f"vs H1 {h1_mean:.3f} ({lift(h1_mean)})   "
              f"vs H2 {h2_mean:.3f} ({lift(h2_mean)})")


if __name__ == "__main__":
    main()