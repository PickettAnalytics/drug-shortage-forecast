"""Tests for the data loader's typing / exclusion / slicing helpers.

These are pure-function tests over small DataFrames — no DuckDB.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from shortage_forecast import config
from shortage_forecast.data_loader import (
    _apply_exclusion,
    _coerce_dtypes,
    _slice_split,
    load_splits,
)


def _minimal_panel(n_dins: int = 3, n_months: int = 6) -> pd.DataFrame:
    """A DataFrame with exactly the columns _coerce_dtypes / _apply_exclusion
    care about. Avoids depending on the demo builder for these unit tests."""
    months = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    rows = []
    for di in range(n_dins):
        for mi, m in enumerate(months):
            row = {
                "observation_date": m,
                "din": f"D{di:03d}",
                "drug_code": f"C{di:03d}",
                config.TARGET: int((di + mi) % 3 == 0),
                config.EXCLUSION_FLAG: bool(mi == 0 and di == 0),
            }
            for c in config.NUMERIC_FEATURES + config.FDA_FEATURES:
                row[c] = float(di + mi)
            for c in config.BOOLEAN_FEATURES:
                row[c] = bool((di + mi) % 2)
            for c in config.CATEGORICAL_FEATURES:
                row[c] = "X"
            rows.append(row)
    return pd.DataFrame(rows)


def test_coerce_dtypes_assigns_expected_types():
    df = _coerce_dtypes(_minimal_panel())

    assert pd.api.types.is_datetime64_any_dtype(df["observation_date"])
    assert df[config.TARGET].dtype == np.int8
    assert df[config.EXCLUSION_FLAG].dtype == bool
    for col in config.BOOLEAN_FEATURES:
        assert df[col].dtype == bool, f"{col} should be bool"
    for col in config.CATEGORICAL_FEATURES:
        assert isinstance(df[col].dtype, pd.CategoricalDtype), (
            f"{col} should be categorical"
        )


def test_apply_exclusion_drops_in_shortage_rows_and_column():
    df = _coerce_dtypes(_minimal_panel())
    n_before = len(df)
    n_excl = int(df[config.EXCLUSION_FLAG].sum())
    out = _apply_exclusion(df)

    assert len(out) == n_before - n_excl
    assert config.EXCLUSION_FLAG not in out.columns


def test_slice_split_is_inclusive_on_both_ends():
    df = _coerce_dtypes(_minimal_panel(n_dins=2, n_months=6))
    sliced = _slice_split(df, date(2020, 2, 1), date(2020, 4, 1))
    months = sliced["observation_date"].dt.to_period("M").unique()
    assert {str(m) for m in months} == {"2020-02", "2020-03", "2020-04"}


def test_load_splits_against_demo_db(demo_db_env):
    """End-to-end smoke test of load_splits against the synthetic panel."""
    train, val, test = load_splits(verbose=False)

    # Each split should be non-empty.
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0

    # No row should carry the exclusion flag (it gets dropped).
    assert config.EXCLUSION_FLAG not in train.columns

    # Temporal ordering is honoured.
    assert train["observation_date"].max() < pd.Timestamp(config.SPLITS.val_start)
    assert val["observation_date"].max() < pd.Timestamp(config.SPLITS.test_start)

    # Every modelling feature is present.
    missing = [c for c in config.FEATURES if c not in train.columns]
    assert missing == [], f"missing features: {missing}"
