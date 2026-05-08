"""Centralised runtime configuration for the drug-shortage-forecast pipeline.

Everything that's a knob — paths, split dates, feature groups, hyperparameters,
monotone constraints — lives here. Other modules import from this file rather
than redeclaring constants, so a single edit propagates through baseline.py,
operational.py, and the test suite.

Two environment variables can override defaults at runtime:

  DRUG_SHORTAGE_DB         absolute or relative path to the DuckDB file the
                           pipeline reads from. Lets `make demo-train` point
                           at the synthetic database without code changes.
  DRUG_SHORTAGE_OUTPUT_DIR directory for trained-model artefacts (CSVs, PNGs).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# src/shortage_forecast/config.py  ->  project root is two parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DB_PATH = PROJECT_ROOT / "drug_shortages.duckdb"
DEMO_DB_PATH = PROJECT_ROOT / "drug_shortages_demo.duckdb"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
STAGING_DIR = PROJECT_ROOT / "data" / "staging"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "baseline_results"

PANEL_TABLE = "main_marts.mrt_shortage_panel"


def get_db_path() -> Path:
    """Resolve the active DuckDB path, honouring DRUG_SHORTAGE_DB."""
    override = os.environ.get("DRUG_SHORTAGE_DB")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_DB_PATH


def get_output_dir() -> Path:
    """Resolve the trained-model artefact dir, honouring DRUG_SHORTAGE_OUTPUT_DIR."""
    override = os.environ.get("DRUG_SHORTAGE_OUTPUT_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_OUTPUT_DIR


# ---------------------------------------------------------------------------
# Labels & split boundaries
# ---------------------------------------------------------------------------

TARGET = "shortage_started_within_90d"
EXCLUSION_FLAG = "was_in_shortage_on_obs_date"


# Split boundaries.
#
# Test cutoff is set so the 90-day forward label is always observable given
# the current shortage data. With source data updated through 2026-04 and
# panel observation_date extending to 2026-01-01, the latest observation
# with a complete label window is 2026-01-01 (label window closes 2026-04).
#
# Buffers between train/val and val/test are 3 months wide because the
# label peeks 90 days forward — without them, the last training month's
# label window overlaps the first validation month.
@dataclass(frozen=True)
class SplitConfig:
    train_start: date = date(2018, 1, 1)
    train_end:   date = date(2023, 12, 1)    # inclusive; 72 months of training
    # buffer:   2024-01-01 .. 2024-03-01   excluded
    val_start:   date = date(2024, 4, 1)
    val_end:     date = date(2024, 12, 1)    # inclusive; 9 months
    # buffer:   2025-01-01 .. 2025-03-01   excluded
    test_start:  date = date(2025, 4, 1)
    test_end:    date = date(2026, 1, 1)     # inclusive; 10 months


SPLITS = SplitConfig()


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
RANDOM_TIEBREAK_SEED = 42

TOP_K_VALUES = [10, 25, 100, 500, 1000]
TOP_K_OPERATIONAL = [10, 25, 50, 100]

# CatBoost hyperparameters — selected from a multi-objective Optuna study
# (`gbm_pk_tuning`) that optimised val per-month blend P@10 + P@25 directly,
# rather than pooled val PR-AUC. The balanced (max-sum) Pareto point on the
# 2024-04..2024-12 val window beats the previous LightGBM production at every
# operational K and on PR-AUC. See `experiments_results_pk/summary.md`.
# Values are the exact Optuna outputs from `experiments_results_pk/best_params.json`;
# rounding them caused a measurable per-month P@10 drift on the test set.
CATBOOST_PARAMS = dict(
    iterations=2000,
    learning_rate=0.08718837518272327,
    depth=8,
    l2_leaf_reg=15.173802128614698,
    random_strength=0.03024369161272648,
    bagging_temperature=0.07728308264433714,
    border_count=122,
)

# Alternative CatBoost hyperparameters tuned for pooled val PR-AUC instead of
# per-month P@K. Trades P@10 (0.530) for higher P@25 (0.436). Swap into
# CATBOOST_PARAMS if the operational priority shifts to deeper monitoring
# lists. Source: `experiments_results/best_params.json`.
CATBOOST_PARAMS_PR_AUC = dict(
    iterations=2000,
    learning_rate=0.02823663521722853,
    depth=8,
    l2_leaf_reg=5.040998330714447,
    random_strength=0.25152823846957467,
    bagging_temperature=0.6129578000958535,
    border_count=41,
)

EARLY_STOPPING_ROUNDS = 100

LOGISTIC_PARAMS = dict(
    solver="liblinear",
    max_iter=1000,
    class_weight="balanced",
)


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    # Drug intrinsic
    "drug_age_years",
    "ingredient_count",
    # Own-DIN shortage history
    "shortages_prior_12m",
    "shortages_prior_36m",
    "shortages_all_prior",
    "days_since_last_shortage",
    "days_since_first_shortage",
    "longest_prior_shortage_days",
    "avg_inter_shortage_interval_days",
    "days_overdue",
    # Manufacturer
    "mfr_portfolio_size",
    "mfr_shortages_prior_12m",
    "mfr_shortage_rate_12m",
    "mfr_shortage_rate_3m",
    "mfr_shortage_rate_delta_3m_vs_12m",
    # Market structure
    "competing_drugs_same_ai_group",
    "mfr_share_of_ai_group",
    "n_manufacturers_in_ai_group",
    # Peer shortage
    "peer_shortages_prior_12m_same_ai_group",
    # Discontinuation
    "peer_discontinuations_prior_12m",
    "peer_discontinuations_prior_36m",
    "days_since_peer_discontinuation",
    "mfr_discontinuations_prior_12m",
    "mfr_discontinuation_rate_12m",
]

# CIHI numeric coverage features (n_jurisdictions_on_formulary, n_programs_as_benefit)
# were trialed and dropped: they correlate strongly with each other and act as a
# generic "popularity" signal that pushes broadly-covered drugs to the top of the
# cold-start ranking, collapsing precision at small K. formulary_is_biologics had
# zero gain. Only formulary_is_generic is retained — it carries the drug-type
# split without the popularity bias.
BOOLEAN_FEATURES = [
    "is_pediatric",
    "has_atc_classification",
    "was_ever_in_shortage",
    "peer_any_in_shortage_now_same_ai_group",
    "formulary_is_generic",
]

CATEGORICAL_FEATURES_LOW_CARD = [
    "schedule",
    "atc_anatomic_group",
]
CATEGORICAL_FEATURES_HIGH_CARD = [
    "primary_route",
    "atc_therapeutic_group",
    "primary_form",
]

CATEGORICAL_FEATURES = CATEGORICAL_FEATURES_LOW_CARD + CATEGORICAL_FEATURES_HIGH_CARD

FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES

# FDA shortage signals — present in the panel but not yet used in the unified
# model. Retained here for future experimentation.
FDA_FEATURES = [
    "fda_ingredient_match_flag",
    "fda_active_ingredients_in_us_shortage",
    "fda_ingredients_in_us_shortage_12m",
]

META_COLS = ["observation_date", "din", "drug_code"]


# ---------------------------------------------------------------------------
# Monotone constraints
# ---------------------------------------------------------------------------
#
# Per-feature monotone constraints encode a domain prior strong enough to pin
# the GBM's gradient direction on its most reliable signals: more recent
# shortage activity must monotonically raise predicted risk, and more days
# since the last shortage must monotonically lower it. This stops the GBM
# from learning interaction shapes that downrank drugs whose shortage history
# alone would already place them near the top — the failure mode we observed
# when per-month P@10 trailed a heuristic that did nothing but sort on
# shortages_prior_12m.

MONOTONE_INCREASING_FEATURES = {
    "shortages_prior_12m",
    "shortages_prior_36m",
    "shortages_all_prior",
    "mfr_shortages_prior_12m",
    "mfr_shortage_rate_12m",
}
MONOTONE_DECREASING_FEATURES = {
    "days_since_last_shortage",
}


def monotone_constraints(feature_cols: list[str]) -> list[int]:
    """Monotone-constraint vector aligned with `feature_cols`.

    +1 for features in MONOTONE_INCREASING_FEATURES, -1 for decreasing,
    0 otherwise. Order matters: the returned list lines up with the column
    order of the training matrix. CatBoost / LightGBM / XGBoost all accept
    this list-of-ints format.
    """
    out: list[int] = []
    for f in feature_cols:
        if f in MONOTONE_INCREASING_FEATURES:
            out.append(1)
        elif f in MONOTONE_DECREASING_FEATURES:
            out.append(-1)
        else:
            out.append(0)
    return out
