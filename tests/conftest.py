"""Shared fixtures for the test suite.

The big-ticket fixture here is `demo_db_path`: a session-scoped synthetic
DuckDB built once via :func:`shortage_forecast.demo.build_demo_database`.
Tests that need a populated panel point ``DRUG_SHORTAGE_DB`` at it before
calling into the modelling code, and unset the env var on teardown so the
real db isn't accidentally referenced.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from shortage_forecast.demo import build_demo_database


@pytest.fixture(scope="session")
def demo_db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a small synthetic panel once per test session."""
    path = tmp_path_factory.mktemp("demo_db") / "demo.duckdb"
    # Smaller than the CLI default — tests just need a working panel.
    build_demo_database(db_path=path, n_dins=120, seed=7)
    return path


@pytest.fixture
def demo_db_env(demo_db_path: Path) -> Iterator[Path]:
    """Set DRUG_SHORTAGE_DB to the demo path for the duration of the test."""
    previous = os.environ.get("DRUG_SHORTAGE_DB")
    os.environ["DRUG_SHORTAGE_DB"] = str(demo_db_path)
    try:
        yield demo_db_path
    finally:
        if previous is None:
            os.environ.pop("DRUG_SHORTAGE_DB", None)
        else:
            os.environ["DRUG_SHORTAGE_DB"] = previous
