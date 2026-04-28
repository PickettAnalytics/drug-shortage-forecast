"""Tests for the centralised config layer.

The point of these tests is to lock in a few invariants that the rest of
the pipeline relies on — the SplitConfig date arithmetic, the monotone
constraint vector aligning with FEATURES, and the env var override for
`get_db_path`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shortage_forecast import config


def test_split_dates_are_in_order_with_buffers():
    s = config.SPLITS
    assert s.train_start < s.train_end < s.val_start < s.val_end < s.test_start < s.test_end
    # Buffer between train_end and val_start must be > 90 days because the
    # forward label peeks 90 days ahead.
    assert (s.val_start - s.train_end).days > 90
    assert (s.test_start - s.val_end).days > 90


def test_features_list_has_no_duplicates():
    assert len(config.FEATURES) == len(set(config.FEATURES))


def test_categorical_features_are_subset_of_features():
    cat = set(config.CATEGORICAL_FEATURES)
    assert cat.issubset(set(config.FEATURES))


def test_monotone_constraints_aligns_with_features():
    constraints = config.monotone_constraints(config.FEATURES)
    assert len(constraints) == len(config.FEATURES)

    # Every increasing feature must map to +1, decreasing to -1, others 0.
    for feat, c in zip(config.FEATURES, constraints):
        if feat in config.MONOTONE_INCREASING_FEATURES:
            assert c == 1, f"{feat} should be +1"
        elif feat in config.MONOTONE_DECREASING_FEATURES:
            assert c == -1, f"{feat} should be -1"
        else:
            assert c == 0, f"{feat} should be 0 (got {c})"


def test_monotone_constraints_increasing_set_is_in_features():
    """Catches typos: an increasing feature name that isn't in FEATURES is
    a silent no-op in LightGBM, which is exactly the kind of bug we want to
    surface in CI."""
    assert config.MONOTONE_INCREASING_FEATURES.issubset(set(config.FEATURES))
    assert config.MONOTONE_DECREASING_FEATURES.issubset(set(config.FEATURES))


def test_get_db_path_respects_env_override(monkeypatch, tmp_path: Path):
    target = tmp_path / "custom.duckdb"
    monkeypatch.setenv("DRUG_SHORTAGE_DB", str(target))
    assert config.get_db_path() == target.resolve()


def test_get_db_path_default_when_unset(monkeypatch):
    monkeypatch.delenv("DRUG_SHORTAGE_DB", raising=False)
    assert config.get_db_path() == config.DEFAULT_DB_PATH


@pytest.mark.parametrize("k", [10, 25, 100])
def test_top_k_values_are_positive(k: int):
    assert k in config.TOP_K_VALUES or k in config.TOP_K_OPERATIONAL
