"""Tests for the model builders + a small end-to-end fit on the demo DB.

The end-to-end test is marked `slow`: it actually fits a logistic regression
plus a tiny LightGBM. CI runs it; local quick iterations can skip via
`pytest -m "not slow"`.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from shortage_forecast import config
from shortage_forecast.baseline import (
    build_features,
    build_lightgbm_model,
    build_logistic_pipeline,
    fit_lightgbm,
    fit_logistic,
    predict_proba,
)
from shortage_forecast.data_loader import load_splits


def test_build_logistic_pipeline_returns_sklearn_pipeline():
    pipe = build_logistic_pipeline()
    assert isinstance(pipe, Pipeline)
    assert "prep" in pipe.named_steps
    assert "lr" in pipe.named_steps


def test_build_lightgbm_model_has_constraints_aligned_with_features():
    """The model's monotone-constraints vector must line up element-by-element
    with FEATURES — otherwise we'd silently constrain the wrong column."""
    model = build_lightgbm_model()
    constraints = model.get_params()["monotone_constraints"]
    assert len(constraints) == len(config.FEATURES)
    expected = config.monotone_constraints(config.FEATURES)
    assert list(constraints) == expected


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
def test_lightgbm_fits_with_small_demo_db(demo_db_env):
    """Fit a tiny LightGBM end-to-end on the synthetic panel.

    We override n_estimators to keep this test fast — we're checking the
    training plumbing works, not headline metrics.
    """
    train, val, _ = load_splits(verbose=False)
    X_train, y_train = build_features(train)
    X_val,   y_val   = build_features(val)

    # Custom small model to keep CI quick — but it must keep the same
    # monotone constraints so we don't silently regress that wiring.
    model = lgb.LGBMClassifier(
        n_estimators=50,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        monotone_constraints=config.monotone_constraints(config.FEATURES),
        monotone_constraints_method="basic",
        random_state=config.RANDOM_STATE,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        categorical_feature=config.CATEGORICAL_FEATURES,
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
    )
    probs = predict_proba(model, X_val)
    assert probs.shape == (len(X_val),)
    assert np.all((probs >= 0) & (probs <= 1))


@pytest.mark.slow
def test_fit_lightgbm_default_params_train_on_demo(demo_db_env):
    """Smoke-test the actual `fit_lightgbm` helper with default params.

    This catches breakages in the kwargs/callbacks plumbing — it's the same
    code path baseline.main() uses, but on the demo DB and without any
    metric assertions.
    """
    train, val, _ = load_splits(verbose=False)
    X_train, y_train = build_features(train)
    X_val,   y_val   = build_features(val)

    model = fit_lightgbm(X_train, y_train, X_val, y_val)
    assert model.best_iteration_ is not None
