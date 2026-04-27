"""
Warm/cold split models for drug shortage prediction.

Splits the panel into two regimes based on prior shortage history:

  Warm-start: shortages_all_prior > 0
    Drug has had at least one shortage strictly before the observation date.
    Rich shortage-history features give strong signal.
    Trained and evaluated on warm rows only.

  Cold-start: shortages_all_prior == 0
    Drug has never had a shortage as of the observation date.
    First-event prediction with no shortage history.
    Uses base features PLUS FDA shortage signals (fda_* columns).

A DIN transitions from cold to warm in the month after its first shortage
starts, so a single DIN may contribute rows to both segments.

Evaluation:
  1. Per-segment: warm model on warm test rows; cold model on cold test rows.
  2. Combined portfolio: stitched scores over the full test set, matching
     the strata from baseline.py for direct comparison.

Output: ./warm_cold_results/

Usage:
    python warm_cold_model.py
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from data_loader import (
    CATEGORICAL_FEATURES,
    COLD_FEATURES,
    TARGET,
    WARM_FEATURES,
    load_splits,
)

OUTPUT_DIR = Path("./warm_cold_results")
TOP_K_VALUES = [100, 500, 1000]
RANDOM_STATE = 42

# WARM_FEATURES and COLD_FEATURES are defined in data_loader.
# Warm: base features minus was_ever_in_shortage (always True for warm rows).
# Cold: base features minus constant shortage-history columns, plus FDA signals.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    n = len(y_true)
    if n < k:
        return float("nan")
    top_idx = np.argpartition(-y_score, k - 1)[:k]
    return float(y_true[top_idx].sum()) / k


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    n = len(y_true)
    n_pos = int(y_true.sum())

    if n_pos == 0 or n_pos == n:
        return {
            "n": n,
            "n_positive": n_pos,
            "base_rate": n_pos / n if n else float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "brier": brier_score_loss(y_true, y_score) if n else float("nan"),
            **{f"precision_at_{k}": float("nan") for k in TOP_K_VALUES},
        }

    metrics = {
        "n": n,
        "n_positive": n_pos,
        "base_rate": n_pos / n,
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
    }
    for k in TOP_K_VALUES:
        metrics[f"precision_at_{k}"] = precision_at_k(y_true, y_score, k)
    return metrics


def format_table(df: pd.DataFrame) -> str:
    fmt = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in ("n", "n_positive"):
            fmt[col] = df[col].map(lambda v: f"{int(v):,}" if pd.notna(v) else "—")
        elif col in ("stratum", "model", "segment"):
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _prep_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def fit_lightgbm(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    label: str = "warm",
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=100,
        random_state=RANDOM_STATE,
        verbose=-1,
    )

    X_train = _prep_categoricals(train[feature_cols])
    X_val = _prep_categoricals(val[feature_cols])

    model.fit(
        X_train,
        train[TARGET],
        eval_set=[(X_val, val[TARGET])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        categorical_feature=CATEGORICAL_FEATURES,
    )
    print(f"  {label} model best iteration: {model.best_iteration_}")
    return model


def predict_proba(model: lgb.LGBMClassifier, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    X = _prep_categoricals(df[feature_cols])
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_segment(
    model: lgb.LGBMClassifier,
    data: pd.DataFrame,
    feature_cols: list[str],
    segment_name: str,
) -> pd.DataFrame:
    """Metrics for one model evaluated on its own segment."""
    if len(data) == 0:
        return pd.DataFrame()

    scores = predict_proba(model, data, feature_cols)
    y_true = data[TARGET].to_numpy()
    m = compute_metrics(y_true, scores)
    m["segment"] = segment_name
    return pd.DataFrame([m])


def evaluate_combined(
    warm_model: lgb.LGBMClassifier,
    cold_model: lgb.LGBMClassifier,
    test: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Stitch warm + cold predictions into a single score array over the full
    test set. Returns (scores_array, metrics_dataframe_by_stratum).
    """
    warm_mask = test["shortages_all_prior"].to_numpy() > 0
    cold_mask = ~warm_mask

    scores = np.empty(len(test), dtype=float)
    scores[warm_mask] = predict_proba(warm_model, test[warm_mask], WARM_FEATURES)
    scores[cold_mask] = predict_proba(cold_model, test[cold_mask], COLD_FEATURES)

    y_true = test[TARGET].to_numpy()

    rows = []
    strata = {
        "overall": np.ones(len(test), dtype=bool),
        "warm_start": warm_mask,
        "cold_start": cold_mask,
        "atc_N": (test["atc_anatomic_group"] == "N").to_numpy(),
        "atc_C": (test["atc_anatomic_group"] == "C").to_numpy(),
        "atc_J": (test["atc_anatomic_group"] == "J").to_numpy(),
        "atc_other": (~test["atc_anatomic_group"].isin({"N", "C", "J"})).to_numpy(),
    }
    for name, mask in strata.items():
        if mask.sum() == 0:
            continue
        m = compute_metrics(y_true[mask], scores[mask])
        m["stratum"] = name
        rows.append(m)

    metrics_df = pd.DataFrame(rows)
    cols = ["stratum", "n", "n_positive", "base_rate",
            "roc_auc", "pr_auc", "brier"] + [f"precision_at_{k}" for k in TOP_K_VALUES]
    return scores, metrics_df[cols]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Load ---
    print("Loading splits...")
    train, val, test = load_splits()

    # --- Segment masks ---
    warm_train = train[train["shortages_all_prior"] > 0].reset_index(drop=True)
    cold_train = train[train["shortages_all_prior"] == 0].reset_index(drop=True)
    warm_val   = val[val["shortages_all_prior"] > 0].reset_index(drop=True)
    cold_val   = val[val["shortages_all_prior"] == 0].reset_index(drop=True)
    warm_test  = test[test["shortages_all_prior"] > 0].reset_index(drop=True)
    cold_test  = test[test["shortages_all_prior"] == 0].reset_index(drop=True)

    print(f"\nSegment sizes:")
    print(f"  train  warm={len(warm_train):,}  cold={len(cold_train):,}")
    print(f"  val    warm={len(warm_val):,}  cold={len(cold_val):,}")
    print(f"  test   warm={len(warm_test):,}  cold={len(cold_test):,}")

    pos_warm_train = int(warm_train[TARGET].sum())
    pos_cold_train = int(cold_train[TARGET].sum())
    print(f"\n  train positives  warm={pos_warm_train:,}  cold={pos_cold_train:,}")

    # --- Fit ---
    print("\nFitting warm-start LightGBM...")
    warm_model = fit_lightgbm(warm_train, warm_val, WARM_FEATURES, label="warm")

    print("Fitting cold-start LightGBM...")
    cold_model = fit_lightgbm(cold_train, cold_val, COLD_FEATURES, label="cold")

    # --- Per-segment evaluation on test ---
    print("\nEvaluating per segment on test set...")
    warm_seg_metrics = evaluate_segment(warm_model, warm_test, WARM_FEATURES, "warm_start")
    cold_seg_metrics = evaluate_segment(cold_model, cold_test, COLD_FEATURES, "cold_start")

    seg_metrics = pd.concat([warm_seg_metrics, cold_seg_metrics], ignore_index=True)
    seg_metrics.to_csv(OUTPUT_DIR / "segment_metrics.csv", index=False)

    print("\n--- Per-segment test metrics ---")
    print(format_table(seg_metrics))

    # --- Combined portfolio evaluation ---
    print("\nEvaluating combined (stitched) scores on full test set...")
    _, combined_metrics = evaluate_combined(warm_model, cold_model, test)
    combined_metrics.to_csv(OUTPUT_DIR / "combined_metrics.csv", index=False)

    print("\n--- Combined test metrics ---")
    print(format_table(combined_metrics))

    # --- Feature importance ---
    _save_importance(warm_model, WARM_FEATURES, "warm", OUTPUT_DIR / "importance_warm.csv")
    _save_importance(cold_model, COLD_FEATURES, "cold", OUTPUT_DIR / "importance_cold.csv")

    print(f"\nArtifacts written to {OUTPUT_DIR.resolve()}")


def _save_importance(
    model: lgb.LGBMClassifier,
    feature_cols: list[str],
    label: str,
    path: Path,
) -> None:
    imp = pd.DataFrame({
        "feature": model.feature_name_,
        "gain": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    imp.to_csv(path, index=False)
    print(f"\nTop-10 {label} features by gain:")
    print(imp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
