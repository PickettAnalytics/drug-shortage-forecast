"""
Data loader for the drug shortage prediction panel.

Single public entry point: `load_splits()` returns train / val / test
DataFrames from main_marts.mrt_shortage_panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd


DEFAULT_DB_PATH = Path(r"E:\Projects\drug-shortage-forecast\drug_shortages.duckdb")
PANEL_TABLE = "main_marts.mrt_shortage_panel"

TARGET = "shortage_started_within_90d"
EXCLUSION_FLAG = "was_in_shortage_on_obs_date"


@dataclass(frozen=True)
class SplitConfig:
    train_start: date = date(2018, 1, 1)
    train_end:   date = date(2023, 6, 1)
    val_start:   date = date(2023, 10, 1)
    val_end:     date = date(2024, 6, 1)
    test_start:  date = date(2024, 10, 1)
    test_end:    date = date(2025, 6, 1)


SPLITS = SplitConfig()


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

BOOLEAN_FEATURES = [
    "is_pediatric",
    "has_atc_classification",
    "was_ever_in_shortage",
    "peer_any_in_shortage_now_same_ai_group",
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

META_COLS = ["observation_date", "din", "drug_code"]


def _load_raw(db_path: Path) -> pd.DataFrame:
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
    df = df.copy()
    df["observation_date"] = pd.to_datetime(df["observation_date"])

    for col in BOOLEAN_FEATURES + [EXCLUSION_FLAG]:
        df[col] = df[col].astype("bool")

    df[TARGET] = df[TARGET].astype("int8")

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _apply_exclusion(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df[EXCLUSION_FLAG]].drop(columns=[EXCLUSION_FLAG]).reset_index(drop=True)


def _slice_split(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    mask = (df["observation_date"] >= start_ts) & (df["observation_date"] <= end_ts)
    return df.loc[mask].reset_index(drop=True)


def load_splits(
    db_path: Path | str = DEFAULT_DB_PATH,
    splits: SplitConfig = SPLITS,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    print(f"  {'Total features:':<36} {len(FEATURES):>9,}")


if __name__ == "__main__":
    train, val, test = load_splits()

    print("\nSanity checks")
    print("-" * 80)

    for name, df in [("train", train), ("val", val), ("test", test)]:
        dups = df.duplicated(subset=["observation_date", "din"]).sum()
        print(f"  {name:<6} duplicate (obs_date, din) rows: {dups}")

    assert train["observation_date"].max() < pd.Timestamp(SPLITS.val_start), \
        "Train/val overlap"
    assert val["observation_date"].max() < pd.Timestamp(SPLITS.test_start), \
        "Val/test overlap"
    print("  train/val/test temporal boundaries clean")

    print(f"  train rows with NaN target: {train[TARGET].isna().sum()}")

    missing = [c for c in FEATURES if c not in train.columns]
    print(f"  features missing from train: {missing if missing else 'none'}")