"""Data loader for the drug shortage prediction panel.

Single public entry point: `load_splits()` returns train / val / test
DataFrames from the dbt-built `mrt_shortage_panel`.

All configuration (paths, split dates, feature groups, target column) is
imported from `shortage_forecast.config` so any downstream change to the
modelling table or feature set lands in one place.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from shortage_forecast.config import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUSION_FLAG,
    FDA_FEATURES,
    FEATURES,
    META_COLS,
    NUMERIC_FEATURES,
    PANEL_TABLE,
    SPLITS,
    TARGET,
    SplitConfig,
    get_db_path,
)

# Re-export for callers that previously imported these names directly from
# `data_loader`. New code should prefer `shortage_forecast.config`.
__all__ = [
    "load_splits",
    "TARGET",
    "EXCLUSION_FLAG",
    "FEATURES",
    "FDA_FEATURES",
    "NUMERIC_FEATURES",
    "BOOLEAN_FEATURES",
    "CATEGORICAL_FEATURES",
    "SplitConfig",
    "SPLITS",
]


def _load_raw(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found at {db_path.resolve()}")

    all_feature_cols = FEATURES + FDA_FEATURES
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        select_cols = ", ".join(META_COLS + [TARGET, EXCLUSION_FLAG] + all_feature_cols)
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

    for col in NUMERIC_FEATURES + FDA_FEATURES:
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
    db_path: Path | str | None = None,
    splits: SplitConfig = SPLITS,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    db_path = Path(db_path) if db_path is not None else get_db_path()

    if verbose:
        print(f"Reading {PANEL_TABLE} from {db_path}")
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
