"""
Data loader for the drug shortage prediction panel.

Single public entry point: `load_splits()` returns train / val / test
DataFrames from main_marts.mrt_shortage_panel, with:

  - rows where the drug is currently in shortage (was_in_shortage_on_obs_date = 1)
    removed from all splits — the target "shortage starts in next 90 days"
    only makes sense for eligible drugs.
  - temporal splits per the EDA decisions (see SPLITS below).
  - 3-month buffers between train/val and val/test so the 90-day forward
    label can't leak across the boundary.
  - dtypes set correctly for downstream modelling (categoricals as category,
    booleans as bool, nullable integers preserved).

Usage:
    from data_loader import load_splits, TARGET, FEATURES

    train, val, test = load_splits()
    X_train = train[FEATURES]
    y_train = train[TARGET]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = Path(r"E:\Projects\drug-shortage-forecast\drug_shortages.duckdb")
PANEL_TABLE = "main_marts.mrt_shortage_panel"

TARGET = "shortage_started_within_90d"
EXCLUSION_FLAG = "was_in_shortage_on_obs_date"

# Split boundaries, derived from the EDA.
#   - Training spans all three temporal regimes (pre-COVID, COVID dip, recovery).
#   - Buffers are 3 months wide because the label peeks 90 days forward;
#     without them, the last training month's label window overlaps the
#     first validation month.
#   - Test is held for a single final evaluation, not for tuning.
@dataclass(frozen=True)
class SplitConfig:
    train_start: date = date(2018, 1, 1)
    train_end:   date = date(2023, 6, 1)    # inclusive
    # buffer:   2023-07-01 .. 2023-09-01   excluded
    val_start:   date = date(2023, 10, 1)
    val_end:     date = date(2024, 6, 1)    # inclusive
    # buffer:   2024-07-01 .. 2024-09-01   excluded
    test_start:  date = date(2024, 10, 1)
    test_end:    date = date(2025, 6, 1)    # inclusive


SPLITS = SplitConfig()


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
# Explicit lists rather than "everything except target" so that schema
# changes don't silently alter the feature set.

NUMERIC_FEATURES = [
    "drug_age_years",
    "ingredient_count",
    "shortages_prior_12m",
    "shortages_prior_36m",
    "shortages_all_prior",
    "days_since_last_shortage",
    "days_since_first_shortage",
    "longest_prior_shortage_days",
    "mfr_portfolio_size",
    "mfr_shortages_prior_12m",
    "mfr_shortage_rate_12m",
    "competing_drugs_same_ai_group",
]

BOOLEAN_FEATURES = [
    "is_pediatric",
    "has_atc_classification",
    "was_ever_in_shortage",
]

# Ordered low to high cardinality. For linear models, target-encode the
# high-cardinality ones; for trees, just pass as category and let the
# splitter handle it.
CATEGORICAL_FEATURES_LOW_CARD = [
    "schedule",              # 13
    "atc_anatomic_group",    # 15
]
CATEGORICAL_FEATURES_HIGH_CARD = [
    "primary_route",         # 71
    "atc_therapeutic_group", # 93
    "primary_form",          # 95
    # atc_code_full (1,881) intentionally excluded — use therapeutic group instead
]

CATEGORICAL_FEATURES = CATEGORICAL_FEATURES_LOW_CARD + CATEGORICAL_FEATURES_HIGH_CARD

FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES

# Columns we want to carry alongside features/target for diagnostics and
# stratified evaluation, but not feed to the model.
META_COLS = ["observation_date", "din", "drug_code"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_raw(db_path: Path) -> pd.DataFrame:
    """Fetch the full panel from DuckDB."""
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found at {db_path.resolve()}")

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        select_cols = ", ".join(META_COLS + [TARGET, EXCLUSION_FLAG] + FEATURES)
        df = con.sql(f"SELECT {select_cols} FROM {PANEL_TABLE}").df()
    finally:
        con.close()
    return df


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set dtypes that survive the groupby/join chain and work with both
    sklearn and gradient-boosted tree libraries.
    """
    df = df.copy()

    # observation_date arrives as datetime64; normalize to date-at-midnight
    # for clean comparisons against SplitConfig values.
    df["observation_date"] = pd.to_datetime(df["observation_date"])

    # Boolean columns sometimes arrive as object dtype depending on DuckDB version
    for col in BOOLEAN_FEATURES + [EXCLUSION_FLAG]:
        df[col] = df[col].astype("bool")

    # Target as small int (not bool) so .mean() gives a rate and sklearn is happy
    df[TARGET] = df[TARGET].astype("int8")

    # Categoricals — keeping nulls as a distinct category is meaningful
    # for atc_* (8.5% null = radiopharmaceuticals and other classification gaps)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

    # Numeric features: leave nullable where the null is meaningful
    # (days_since_last_shortage etc.), cast the rest to float64
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _apply_exclusion(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where the drug is already in a shortage on the observation date."""
    return df[~df[EXCLUSION_FLAG]].drop(columns=[EXCLUSION_FLAG]).reset_index(drop=True)


def _slice_split(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Inclusive range on observation_date."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    mask = (df["observation_date"] >= start_ts) & (df["observation_date"] <= end_ts)
    return df.loc[mask].reset_index(drop=True)


def load_splits(
    db_path: Path | str = DEFAULT_DB_PATH,
    splits: SplitConfig = SPLITS,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the panel, apply the currently-in-shortage exclusion, and return
    (train, val, test) DataFrames according to `splits`.

    Each returned frame contains META_COLS + [TARGET] + FEATURES.
    The exclusion flag column is dropped before returning.
    """
    db_path = Path(db_path)

    raw = _load_raw(db_path)
    typed = _coerce_dtypes(raw)
    eligible = _apply_exclusion(typed)

    train = _slice_split(eligible, splits.train_start, splits.train_end)
    val   = _slice_split(eligible, splits.val_start,   splits.val_end)
    test  = _slice_split(eligible, splits.test_start,  splits.test_end)

    if verbose:
        _print_split_summary(eligible, train, val, test, splits)

    return train, val, test


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_split_summary(
    eligible: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    splits: SplitConfig,
) -> None:
    def row(name, df, rng):
        n = len(df)
        pos = int(df[TARGET].sum())
        rate = pos / n if n else 0.0
        dins = df["din"].nunique()
        print(f"  {name:<6} {rng:<28} n={n:>9,}  positives={pos:>6,}  "
              f"rate={rate:.4f}  dins={dins:>6,}")

    total_panel_rows = len(eligible)
    used_rows = len(train) + len(val) + len(test)
    print("Split summary")
    print("-" * 80)
    row("train", train, f"{splits.train_start} .. {splits.train_end}")
    row("val",   val,   f"{splits.val_start} .. {splits.val_end}")
    row("test",  test,  f"{splits.test_start} .. {splits.test_end}")
    print(f"  {'Eligible panel (post-exclusion):':<36} {total_panel_rows:>9,}")
    print(f"  {'Rows used across train/val/test:':<36} {used_rows:>9,} "
          f"({used_rows/total_panel_rows:.1%})")
    print(f"  {'Rows in buffers / unused tail:':<36} "
          f"{total_panel_rows - used_rows:>9,}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train, val, test = load_splits()

    print("\nSanity checks")
    print("-" * 80)

    # Grain
    for name, df in [("train", train), ("val", val), ("test", test)]:
        dups = df.duplicated(subset=["observation_date", "din"]).sum()
        print(f"  {name:<6} duplicate (obs_date, din) rows: {dups}")

    # No temporal overlap
    assert train["observation_date"].max() < pd.Timestamp(SPLITS.val_start), \
        "Train/val overlap"
    assert val["observation_date"].max() < pd.Timestamp(SPLITS.test_start), \
        "Val/test overlap"
    print("  train/val/test temporal boundaries clean")

    # No exclusion-flag rows survived
    # (flag column was dropped, so re-check via the panel's null-free features)
    print(f"  train rows with NaN target: {train[TARGET].isna().sum()}")

    # Feature coverage
    missing = [c for c in FEATURES if c not in train.columns]
    print(f"  features missing from train: {missing if missing else 'none'}")

    print("\nFirst 3 rows of train:")
    print(train.head(3).to_string(index=False))
