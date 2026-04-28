"""
Baseline models for drug shortage prediction.

Trains two models side-by-side on the same splits from `data_loader`:

  1. Logistic regression — simple, calibrated, interpretable floor.
  2. LightGBM — near-SOTA tabular, the number any complex model must beat.

Evaluates both with the metric hierarchy for use case (a) — early-warning
ranking system:

  Primary:    precision@K for K in {100, 500, 1000}   (operational metric)
  Secondary:  PR-AUC (average precision)               (honest under imbalance)
  Tertiary:   ROC-AUC                                  (familiar to audience)
  Diagnostic: Brier score + reliability diagram        (calibration check)

Every metric is reported overall and stratified by:
  - warm-start vs cold-start (was_ever_in_shortage)
  - DIN seen in training vs not
  - ATC anatomic group (N / C / J / Other)
  - observation month (drift within the evaluation window)

The script is organised around four top-level functions — `load_data`,
`build_features`, `train_model`, `evaluate_model` — orchestrated by
`main`. `operational_metrics.py` reuses `build_features` and
`train_model` directly so the two scripts always train identical models.

Usage:
    python baseline.py

Output goes to ./baseline_results/ as CSV + PNGs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
)
from sklearn.calibration import calibration_curve

import lightgbm as lgb

from data_loader import (
    load_splits,
    TARGET,
    NUMERIC_FEATURES,
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORICAL_FEATURES_LOW_CARD,
    CATEGORICAL_FEATURES_HIGH_CARD,
    FEATURES,
)


OUTPUT_DIR = Path("./baseline_results")
TOP_K_VALUES = [10, 25, 100, 500, 1000]
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading & feature prep
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test panel splits from the dbt mart via data_loader."""
    print("Loading panel splits from DuckDB...")
    train, val, test = load_splits()
    print(
        f"  shapes: train={train.shape}  val={val.shape}  test={test.shape}"
    )
    return train, val, test


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract the modelling feature matrix and target from a panel split.

    Casts categorical columns to pandas `category` dtype so LightGBM can
    treat them natively. The feature list itself is defined in
    `data_loader.FEATURES` and intentionally not modified here.
    """
    X = df[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
    y = df[TARGET].to_numpy()
    return X, y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Fraction of positives among the top-k predicted scores.

    If k >= len(y), returns base rate. If fewer than k items exist
    (stratified subset too small), returns NaN.
    """
    n = len(y_true)
    if n < k:
        return float("nan")
    top_idx = np.argpartition(-y_score, k - 1)[:k]
    return float(y_true[top_idx].sum()) / k


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    top_k_values: Iterable[int] = TOP_K_VALUES,
) -> dict:
    """All metrics for one (y_true, y_score) pair."""
    n = len(y_true)
    n_pos = int(y_true.sum())

    # Ranking metrics are undefined if only one class present
    if n_pos == 0 or n_pos == n:
        return {
            "n": n,
            "n_positive": n_pos,
            "base_rate": n_pos / n if n else float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "brier": brier_score_loss(y_true, y_score) if n else float("nan"),
            **{f"precision_at_{k}": float("nan") for k in top_k_values},
        }

    metrics = {
        "n": n,
        "n_positive": n_pos,
        "base_rate": n_pos / n,
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
    }
    for k in top_k_values:
        metrics[f"precision_at_{k}"] = precision_at_k(y_true, y_score, k)
    return metrics


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_logistic_pipeline() -> Pipeline:
    """Logistic regression with skew-aware preprocessing.

    - Numeric features: impute with median, then standard-scale.
      Recency features' nulls get median too (imperfect but simple).
    - Boolean features: passed through as 0/1.
    - Low-card categoricals: one-hot.
    - High-card categoricals: one-hot as well (sparse, no target encoding yet).

    Note: we deliberately DON'T log1p-transform skewed counts. LightGBM
    doesn't need it and the linear model's weakness on skewed features
    is part of the comparison — we want to see how much it matters.
    """
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
            ("cat_low", categorical_pipeline, CATEGORICAL_FEATURES_LOW_CARD),
            ("cat_high", categorical_pipeline, CATEGORICAL_FEATURES_HIGH_CARD),
        ],
        remainder="drop",
    )

    # class_weight='balanced' so the 3% minority doesn't get ignored
    model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline([
        ("prep", preprocessor),
        ("lr", model),
    ])


# Per-feature monotone constraints encode a domain prior strong enough to
# pin the model's gradient direction on its most reliable signals: more
# recent shortage activity must monotonically raise predicted risk, and
# more days since the last shortage must monotonically lower it. This
# stops the GBM from learning interaction shapes that downrank drugs
# whose shortage history alone would already place them near the top —
# the failure mode we observed when per-month P@10 trailed a heuristic
# that did nothing but sort on shortages_prior_12m.
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


def _monotone_constraints(feature_cols: list[str]) -> list[int]:
    out = []
    for f in feature_cols:
        if f in MONOTONE_INCREASING_FEATURES:
            out.append(1)
        elif f in MONOTONE_DECREASING_FEATURES:
            out.append(-1)
        else:
            out.append(0)
    return out


def build_lightgbm_model() -> lgb.LGBMClassifier:
    """LightGBM binary classifier with monotone constraints on shortage signals.

    Constraints (see MONOTONE_INCREASING_FEATURES / MONOTONE_DECREASING_FEATURES)
    pin the model's gradient direction on the features the heuristic relies on;
    the GBM is free to learn anything it likes off the remaining features.
    """
    return lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=25,
        min_child_samples=150,
        monotone_constraints=_monotone_constraints(FEATURES),
        monotone_constraints_method="basic",
        random_state=RANDOM_STATE,
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def fit_logistic(X_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    """Fit the logistic-regression pipeline on a feature matrix."""
    pipe = build_logistic_pipeline()
    pipe.fit(X_train, y_train)
    return pipe


def fit_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
) -> lgb.LGBMClassifier:
    """Fit LightGBM. `(X_val, y_val)` enables early stopping on PR-AUC."""
    model = build_lightgbm_model()
    fit_kwargs: dict = dict(
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        categorical_feature=CATEGORICAL_FEATURES,
    )
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
) -> dict:
    """Train both baselines side-by-side and return them in a dict.

    LightGBM uses `(X_val, y_val)` for early stopping when supplied;
    logistic regression has no early stopping and ignores it.
    """
    n_features = X_train.shape[1]
    pos_rate = float(np.mean(y_train))
    print(
        f"Training start | rows={len(X_train):,}  features={n_features}  "
        f"positive_rate={pos_rate:.4f}"
    )
    t0 = time.perf_counter()

    print("  Fitting logistic regression...")
    lr = fit_logistic(X_train, y_train)

    print("  Fitting LightGBM...")
    gbm = fit_lightgbm(X_train, y_train, X_val, y_val)
    print(f"  LightGBM best iteration: {gbm.best_iteration_}")

    elapsed = time.perf_counter() - t0
    print(f"Training complete in {elapsed:.1f}s.")
    return {"logistic": lr, "lightgbm": gbm}


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Positive-class probability for either model type."""
    X = X.copy()
    # LightGBM needs category dtype; sklearn pipeline doesn't care
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return model.predict_proba(X[FEATURES])[:, 1]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X: pd.DataFrame, y: np.ndarray) -> dict:
    """Compute the headline metrics for `model` on `(X, y)`.

    Returns the same dict shape as `compute_metrics`. For stratified
    breakdowns and per-month drift, see `evaluate_all_strata` and
    `evaluate_monthly_drift`.
    """
    scores = predict_proba(model, X)
    return compute_metrics(np.asarray(y), scores)


@dataclass
class StratifiedSlice:
    name: str
    mask: np.ndarray  # boolean array over the evaluation frame


def build_strata(
    eval_df: pd.DataFrame,
    train_dins: set,
) -> list[StratifiedSlice]:
    """Build the stratification slices from the EDA decisions."""
    slices: list[StratifiedSlice] = []

    # Overall (always first)
    slices.append(StratifiedSlice(
        name="overall",
        mask=np.ones(len(eval_df), dtype=bool),
    ))

    # Warm-start vs cold-start via was_ever_in_shortage
    slices.append(StratifiedSlice(
        name="warm_start (was_ever_in_shortage=True)",
        mask=eval_df["was_ever_in_shortage"].to_numpy(),
    ))
    slices.append(StratifiedSlice(
        name="cold_start (was_ever_in_shortage=False)",
        mask=~eval_df["was_ever_in_shortage"].to_numpy(),
    ))

    # DIN seen in training vs not
    seen_mask = eval_df["din"].isin(train_dins).to_numpy()
    slices.append(StratifiedSlice(
        name="din_seen_in_train",
        mask=seen_mask,
    ))
    slices.append(StratifiedSlice(
        name="din_unseen_in_train",
        mask=~seen_mask,
    ))

    # ATC anatomic group — top 3 + Other
    top_atc = {"N", "C", "J"}
    for group in ["N", "C", "J"]:
        slices.append(StratifiedSlice(
            name=f"atc_{group}",
            mask=(eval_df["atc_anatomic_group"] == group).to_numpy(),
        ))
    slices.append(StratifiedSlice(
        name="atc_other",
        mask=(~eval_df["atc_anatomic_group"].isin(top_atc)).to_numpy(),
    ))

    return slices


def evaluate_all_strata(
    eval_df: pd.DataFrame,
    y_score: np.ndarray,
    strata: list[StratifiedSlice],
) -> pd.DataFrame:
    """Run compute_metrics on each slice, return a tidy DataFrame."""
    y_true = eval_df[TARGET].to_numpy()
    rows = []
    for sl in strata:
        if sl.mask.sum() == 0:
            continue
        m = compute_metrics(y_true[sl.mask], y_score[sl.mask])
        m["stratum"] = sl.name
        rows.append(m)
    df = pd.DataFrame(rows)
    cols = ["stratum", "n", "n_positive", "base_rate",
            "roc_auc", "pr_auc", "brier"] + [f"precision_at_{k}" for k in TOP_K_VALUES]
    return df[cols]


def evaluate_monthly_drift(
    eval_df: pd.DataFrame,
    y_score: np.ndarray,
) -> pd.DataFrame:
    """Per-month metrics within the evaluation window, to visualise drift."""
    df = eval_df.copy()
    df["score"] = y_score
    rows = []
    for obs_date, group in df.groupby("observation_date"):
        m = compute_metrics(
            group[TARGET].to_numpy(),
            group["score"].to_numpy(),
        )
        m["observation_date"] = obs_date
        rows.append(m)
    return pd.DataFrame(rows).sort_values("observation_date")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_reliability(
    results_by_model: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    """Reliability diagram for all models on one chart."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    for name, (y_true, y_score) in results_by_model.items():
        frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=20, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram — test set")
    ax.legend()
    ax.set_xlim(0, max(0.3, ax.get_xlim()[1]))
    ax.set_ylim(0, max(0.3, ax.get_ylim()[1]))
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_monthly_pr_auc(
    monthly_by_model: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """PR-AUC per month across the evaluation window."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, df in monthly_by_model.items():
        ax.plot(df["observation_date"], df["pr_auc"], marker=".", label=name)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Monthly PR-AUC across evaluation window")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_feature_importance_lgbm(
    model: lgb.LGBMClassifier,
    out_path: Path,
    top_n: int = 20,
) -> None:
    """Gain-based importance. SHAP would be better but is slower; save for later."""
    importances = pd.DataFrame({
        "feature": model.feature_name_,
        "gain": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(importances))))
    ax.barh(importances["feature"], importances["gain"])
    ax.set_xlabel("Total gain")
    ax.set_title(f"LightGBM feature importance (top {top_n})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def format_table(df: pd.DataFrame) -> str:
    """Pretty print a metrics DataFrame without mutating the input."""
    fmt = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in ("n", "n_positive"):
            fmt[col] = df[col].map(
                lambda v: f"{int(v):,}" if pd.notna(v) else "—"
            )
        elif col in ("stratum", "observation_date", "model"):
            fmt[col] = df[col].astype(str)
        else:
            def _fmt(v):
                if pd.isna(v):
                    return "—"
                try:
                    return f"{float(v):.4f}"
                except (ValueError, TypeError):
                    return str(v)
            fmt[col] = df[col].map(_fmt)
    return fmt.to_string(index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Load
    train, val, test = load_data()
    train_dins = set(train["din"].unique())

    # 2. Build feature matrices for each split
    X_train, y_train = build_features(train)
    X_val,   y_val   = build_features(val)
    X_test,  y_test  = build_features(test)

    # 3. Train both baselines
    models = train_model(X_train, y_train, X_val, y_val)

    # 4. Score test set + stratified evaluation
    print("\nScoring test set...")
    scores = {name: predict_proba(m, X_test) for name, m in models.items()}

    strata = build_strata(test, train_dins)

    print("\n" + "=" * 80)
    print("STRATIFIED METRICS — TEST SET")
    print("=" * 80)

    strat_dfs = []
    for name, s in scores.items():
        df = evaluate_all_strata(test, s, strata)
        df.insert(0, "model", name)
        strat_dfs.append(df)
        print(f"\n--- {name} ---")
        print(format_table(df.drop(columns=["model"])))
    pd.concat(strat_dfs, ignore_index=True).to_csv(
        OUTPUT_DIR / "stratified_metrics.csv", index=False
    )

    # 5. Monthly drift within test window
    print("\nComputing monthly drift...")
    monthly = {name: evaluate_monthly_drift(test, s) for name, s in scores.items()}
    for name, df in monthly.items():
        df.to_csv(OUTPUT_DIR / f"monthly_{name}.csv", index=False)

    # 6. Plots
    print("Saving plots...")
    plot_reliability(
        {name: (y_test, s) for name, s in scores.items()},
        OUTPUT_DIR / "reliability.png",
    )
    plot_monthly_pr_auc(monthly, OUTPUT_DIR / "monthly_pr_auc.png")
    plot_feature_importance_lgbm(
        models["lightgbm"], OUTPUT_DIR / "lightgbm_importance.png"
    )

    # 7. Headline summary — overall metrics via evaluate_model
    print("\n" + "=" * 80)
    print("HEADLINE — overall test metrics")
    print("=" * 80)
    headline_rows = []
    for name, m in models.items():
        row = evaluate_model(m, X_test, y_test)
        row["model"] = name
        headline_rows.append(row)
    headline = pd.DataFrame(headline_rows)
    cols = ["model", "n", "n_positive", "base_rate",
            "roc_auc", "pr_auc", "brier"] + [f"precision_at_{k}" for k in TOP_K_VALUES]
    headline = headline[cols]
    print(format_table(headline))

    print(f"\nArtifacts written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
