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

Usage:
    python baseline.py

Output goes to ./baseline_results/ as CSV + PNGs.
"""

from __future__ import annotations

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
TOP_K_VALUES = [100, 500, 1000]
RANDOM_STATE = 42


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
# Models
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


def build_lightgbm_model() -> lgb.LGBMClassifier:
    """LightGBM with sensible defaults, no tuning yet.

    Key decisions:
    - is_unbalance=True lets LightGBM up-weight the minority class internally
    - n_estimators=500 with early stopping on validation
    - Categoricals passed via dtype, LightGBM handles them natively
    """
    return lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=100,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


def fit_logistic(train: pd.DataFrame, val: pd.DataFrame) -> Pipeline:
    pipe = build_logistic_pipeline()
    # Logistic regression has no early stopping; ignore val here
    pipe.fit(train[FEATURES], train[TARGET])
    return pipe


def fit_lightgbm(train: pd.DataFrame, val: pd.DataFrame) -> lgb.LGBMClassifier:
    model = build_lightgbm_model()
    # Cast categoricals explicitly so LightGBM knows to treat them natively
    X_train = train[FEATURES].copy()
    X_val = val[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")

    model.fit(
        X_train, train[TARGET],
        eval_set=[(X_val, val[TARGET])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        categorical_feature=CATEGORICAL_FEATURES,
    )
    return model


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Positive-class probability for either model type."""
    X = X.copy()
    # LightGBM needs category dtype; sklearn pipeline doesn't care
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return model.predict_proba(X[FEATURES])[:, 1]


# ---------------------------------------------------------------------------
# Stratified evaluation
# ---------------------------------------------------------------------------

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

    # --- Load ---
    train, val, test = load_splits()
    train_dins = set(train["din"].unique())

    # --- Fit both models on train, validate on val (LightGBM uses val for early stopping) ---
    print("\nFitting logistic regression...")
    lr = fit_logistic(train, val)

    print("Fitting LightGBM...")
    gbm = fit_lightgbm(train, val)
    print(f"LightGBM best iteration: {gbm.best_iteration_}")

    # --- Predict on test ---
    print("\nScoring test set...")
    lr_scores = predict_proba(lr, test)
    gbm_scores = predict_proba(gbm, test)

    # --- Evaluate overall and stratified ---
    strata = build_strata(test, train_dins)

    print("\n" + "=" * 80)
    print("STRATIFIED METRICS — TEST SET")
    print("=" * 80)

    lr_strat = evaluate_all_strata(test, lr_scores, strata)
    lr_strat.insert(0, "model", "logistic")
    gbm_strat = evaluate_all_strata(test, gbm_scores, strata)
    gbm_strat.insert(0, "model", "lightgbm")

    combined = pd.concat([lr_strat, gbm_strat], ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "stratified_metrics.csv", index=False)

    print("\n--- Logistic regression ---")
    print(format_table(lr_strat.drop(columns=["model"])))
    print("\n--- LightGBM ---")
    print(format_table(gbm_strat.drop(columns=["model"])))

    # --- Monthly drift within test window ---
    print("\nComputing monthly drift...")
    lr_monthly = evaluate_monthly_drift(test, lr_scores)
    gbm_monthly = evaluate_monthly_drift(test, gbm_scores)
    lr_monthly.to_csv(OUTPUT_DIR / "monthly_logistic.csv", index=False)
    gbm_monthly.to_csv(OUTPUT_DIR / "monthly_lightgbm.csv", index=False)

    # --- Plots ---
    print("\nSaving plots...")
    y_test = test[TARGET].to_numpy()
    plot_reliability(
        {"logistic": (y_test, lr_scores), "lightgbm": (y_test, gbm_scores)},
        OUTPUT_DIR / "reliability.png",
    )
    plot_monthly_pr_auc(
        {"logistic": lr_monthly, "lightgbm": gbm_monthly},
        OUTPUT_DIR / "monthly_pr_auc.png",
    )
    plot_feature_importance_lgbm(gbm, OUTPUT_DIR / "lightgbm_importance.png")

    # --- Headline summary ---
    print("\n" + "=" * 80)
    print("HEADLINE — overall test metrics")
    print("=" * 80)
    headline_lr  = lr_strat[lr_strat["stratum"] == "overall"].drop(columns=["stratum"])
    headline_gbm = gbm_strat[gbm_strat["stratum"] == "overall"].drop(columns=["stratum"])
    print("\nLogistic:")
    print(format_table(headline_lr))
    print("\nLightGBM:")
    print(format_table(headline_gbm))

    print(f"\nArtifacts written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
