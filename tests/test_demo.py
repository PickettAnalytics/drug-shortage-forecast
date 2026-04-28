"""Tests for the synthetic demo database builder."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from shortage_forecast import config
from shortage_forecast.demo import build_demo_database, build_synthetic_panel


def test_synthetic_panel_has_all_required_columns():
    df = build_synthetic_panel(n_dins=20, seed=1)
    expected = (
        set(config.META_COLS)
        | {config.TARGET, config.EXCLUSION_FLAG}
        | set(config.NUMERIC_FEATURES)
        | set(config.BOOLEAN_FEATURES)
        | set(config.CATEGORICAL_FEATURES)
        | set(config.FDA_FEATURES)
    )
    assert expected.issubset(set(df.columns))


def test_synthetic_panel_target_is_binary():
    df = build_synthetic_panel(n_dins=30, seed=1)
    assert set(df[config.TARGET].unique()).issubset({0, 1})


def test_synthetic_panel_target_rate_is_in_realistic_range():
    """Hand-rolled distributions can drift; this guards against accidental
    regressions where the demo target rate becomes degenerate (0 or 1)."""
    df = build_synthetic_panel(n_dins=200, seed=1)
    rate = df[config.TARGET].mean()
    assert 0.001 < rate < 0.30, f"Demo target rate {rate} is out of band"


def test_build_demo_database_writes_to_panel_table(tmp_path: Path):
    db_path = tmp_path / "demo.duckdb"
    build_demo_database(db_path=db_path, n_dins=30, seed=1)
    assert db_path.exists()

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.sql(f"SELECT COUNT(*) FROM {config.PANEL_TABLE}").fetchone()[0]
    finally:
        con.close()
    assert rows > 0


def test_build_demo_database_is_deterministic(tmp_path: Path):
    """Same seed -> same DataFrame. We round-trip via DuckDB to make sure
    the storage step doesn't reorder rows."""
    a = build_synthetic_panel(n_dins=20, seed=42)
    b = build_synthetic_panel(n_dins=20, seed=42)
    pd.testing.assert_frame_equal(a, b)
