"""Tests for the model builders + a small end-to-end fit on the demo DB.

The end-to-end test is marked `slow`: it actually fits a logistic regression
plus a tiny CatBoost. CI runs it; local quick iterations can skip via
`pytest -m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

from shortage_forecast import config
from shortage_forecast.baseline import (
    _catboost_monotone_dict,
    build_catboost_model,
    build_features,
    build_logistic_pipeline,
    fit_catboost,
    fit_logistic,
    predict_proba,
)
from shortage_forecast.data_loader import load_splits


def test_build_logistic_pipeline_returns_sklearn_pipeline():
    pipe = build_logistic_pipeline()
    assert isinstance(pipe, Pipeline)
    assert "prep" in pipe.named_steps
    assert "lr" in pipe.named_steps


def test_build_catboost_model_has_constraints_aligned_with_features():
    """The model's monotone-constraints dict must contain exactly the
    features in MONOTONE_INCREASING/DECREASING_FEATURES with the right
    sign — otherwise we'd silently constrain the wrong column."""
    model = build_catboost_model()
    assert isinstance(model, CatBoostClassifier)
    constraints = _catboost_monotone_dict()
    for f in config.MONOTONE_INCREASING_FEATURES:
        assert constraints[f] == 1, f"{f} should be +1"
    for f in config.MONOTONE_DECREASING_FEATURES:
        assert constraints[f] == -1, f"{f} should be -1"
    # Zero-constrained features must NOT appear in the dict (the dict only
    # records the constrained ones)
    constrained = config.MONOTONE_INCREASING_FEATURES | config.MONOTONE_DECREASING_FEATURES
    assert set(constraints.keys()) == set(constrained)


def test_build_features_casts_categoricals(demo_db_env):
    train, _, _ = load_splits(verbose=False)
    X, y = build_features(train)

    assert list(X.columns) == config.FEATURES
    for col in config.CATEGORICAL_FEATURES:
        assert isinstance(X[col].dtype, pd.CategoricalDtype)
    assert y.dtype.kind in {"i", "u"}


@pytest.mark.slow
def test_logistic_fits_and_predicts_in_unit_range(demo_db_env):
    train, _, test = load_splits(verbose=False)
    X_train, y_train = build_features(train)
    X_test,  _y_test = build_features(test)

    model = fit_logistic(X_train, y_train)
    probs = predict_proba(model, X_test)
    assert probs.shape == (len(X_test),)
    assert np.all((probs >= 0) & (probs <= 1))


@pytest.mark.slow
def test_catboost_fits_with_small_demo_db(demo_db_env):
    """Fit a tiny CatBoost end-to-end on the synthetic panel.

    We use a hand-rolled small model rather than `fit_catboost` to keep
    this test fast, but it must keep the same monotone constraints so we
    don't silently regress that wiring.
    """
    from catboost import Pool

    train, val, _ = load_splits(verbose=False)
    X_train, y_train = build_features(train)
    X_val,   y_val   = build_features(val)

    # CatBoost can't take pandas Category dtype; reuse the production helper
    from shortage_forecast.baseline import _prepare_for_catboost

    X_train_cb = _prepare_for_catboost(X_train)
    X_val_cb   = _prepare_for_catboost(X_val)

    model = CatBoostClassifier(
        iterations=50,
        learning_rate=0.1,
        depth=4,
        monotone_constraints=_catboost_monotone_dict(),
        eval_metric="PRAUC",
        random_seed=config.RANDOM_STATE,
        early_stopping_rounds=10,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        Pool(X_train_cb, label=y_train, cat_features=config.CATEGORICAL_FEATURES),
        eval_set=Pool(X_val_cb, label=y_val, cat_features=config.CATEGORICAL_FEATURES),
    )
    probs = predict_proba(model, X_val)
    assert probs.shape == (len(X_val),)
    assert np.all((probs >= 0) & (probs <= 1))


@pytest.mark.slow
def test_fit_catboost_default_params_train_on_demo(demo_db_env):
    """Smoke-test the actual `fit_catboost` helper with default params.

    This catches breakages in the kwargs/eval_set plumbing — it's the same
    code path baseline.main() uses, but on the demo DB and without any
    metric assertions.
    """
    train, val, _ = load_splits(verbose=False)
    X_train, y_train = build_features(train)
    X_val,   y_val   = build_features(val)

    model = fit_catboost(X_train, y_train, X_val, y_val)
    assert model.get_best_iteration() is not None
