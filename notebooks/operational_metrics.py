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

    # --- Score four rankers on test ---
    gbm_scores    = predict_proba(gbm, test)
    lr_scores     = predict_proba(lr,  test)
    heur_1_scores = score_heuristic_single(test)
    heur_2_scores = score_heuristic_compound(test)

    monthly_gbm = per_month_metrics(test, gbm_scores,    "lightgbm")
    monthly_lr  = per_month_metrics(test, lr_scores,     "logistic")
    monthly_h1  = per_month_metrics(test, heur_1_scores, "heuristic_single")
    monthly_h2  = per_month_metrics(test, heur_2_scores, "heuristic_compound")

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
    ]:
        merged = pd.merge(
            merged,
            df[["observation_date", "hits_at_10"]].rename(columns={"hits_at_10": name}),
            on="observation_date",
        )
    for col in ["gbm", "logistic", "heur_1", "heur_2"]:
        merged[col] = merged[col].astype(int).astype(str) + "/10"
    print(merged.to_string(index=False))

    # --- Summaries ---
    print("\n" + "=" * 100)
    print(f"SUMMARY — per-month Precision@K (mean base rate = {mean_base:.4f})")
    print("=" * 100)

    summary_gbm = summarize(monthly_gbm, mean_base)
    summary_lr  = summarize(monthly_lr,  mean_base)
    summary_h1  = summarize(monthly_h1,  mean_base)
    summary_h2  = summarize(monthly_h2,  mean_base)

    print_summary_table(summary_gbm, "LightGBM (baseline.py)")
    print_summary_table(summary_lr,  "Logistic regression (baseline.py)")
    print_summary_table(summary_h1,  "Heuristic 1: shortages_prior_12m only")
    print_summary_table(summary_h2,  "Heuristic 2: shortages_prior_12m + mfr_rate tiebreak")

    # --- Headline comparisons ---
    print("\n" + "=" * 100)
    print("HEADLINE — LightGBM vs logistic vs heuristics")
    print("=" * 100)
    for k in TOP_K_OPERATIONAL:
        gbm_mean = summary_gbm.loc[summary_gbm["k"] == k, "mean_precision"].iloc[0]
        lr_mean  = summary_lr.loc[summary_lr["k"]   == k, "mean_precision"].iloc[0]
        h1_mean  = summary_h1.loc[summary_h1["k"]   == k, "mean_precision"].iloc[0]
        h2_mean  = summary_h2.loc[summary_h2["k"]   == k, "mean_precision"].iloc[0]

        def lift(target: float) -> str:
            if target <= 0:
                return "  n/a"
            return f"{(gbm_mean - target) / target:+.1%}"

        print(f"  K={k:<4}  LightGBM {gbm_mean:.3f}   "
              f"vs LR {lr_mean:.3f} ({lift(lr_mean)})   "
              f"vs H1 {h1_mean:.3f} ({lift(h1_mean)})   "
              f"vs H2 {h2_mean:.3f} ({lift(h2_mean)})")


if __name__ == "__main__":
    main()