"""Build a synthetic ``mrt_shortage_panel`` so the modelling pipeline runs
end-to-end without the real raw data.

The synthetic panel mimics the shape, dtypes, and column set of the real
dbt mart at `main_marts.mrt_shortage_panel`. Distributions are coarse
imitations of the real data — enough that monotone constraints, rank
blending, and stratified evaluation all behave on the demo set the same
way they do on real data — but values are random and not interpretable.

The output DuckDB file is written to ``drug_shortages_demo.duckdb`` (or
the path passed to :func:`build_demo_database`). Pointing the modelling
pipeline at it is one env var:

    DRUG_SHORTAGE_DB=drug_shortages_demo.duckdb python -m shortage_forecast.baseline

CLI:

    python -m shortage_forecast.demo                    # build with defaults
    python -m shortage_forecast.demo --n-dins 1000 ...  # custom size
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from shortage_forecast.config import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    DEMO_DB_PATH,
    EXCLUSION_FLAG,
    FDA_FEATURES,
    META_COLS,
    NUMERIC_FEATURES,
    PANEL_TABLE,
    SPLITS,
    TARGET,
)

# Vocabularies used to populate the categorical columns. Real values would
# be the actual ATC code, route, etc; here we just pick from a fixed set so
# the model has variation to learn from.
SCHEDULES = ["Prescription", "OTC", "Schedule II", "Narcotic", "Targeted"]
ATC_ANATOMIC = ["A", "B", "C", "D", "G", "H", "J", "L", "M", "N", "P", "R", "S", "V"]
ATC_THERAPEUTIC = [f"{a}0{i}" for a in ATC_ANATOMIC for i in range(1, 5)]
ROUTES = ["ORAL", "INJECTION", "TOPICAL", "INHALATION", "OPHTHALMIC", "OTIC", "RECTAL"]
FORMS = ["TABLET", "CAPSULE", "SOLUTION", "INJECTABLE", "CREAM", "OINTMENT", "SUSPENSION"]


def _month_starts(start: date, end: date) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="MS")


def build_synthetic_panel(
    n_dins: int = 600,
    start: date = SPLITS.train_start,
    end: date = SPLITS.test_end,
    base_shortage_rate: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a synthetic panel DataFrame matching the real mart's columns.

    Includes META_COLS, TARGET, EXCLUSION_FLAG, every name in FEATURES, and
    every name in FDA_FEATURES. Categorical columns are stored as plain
    strings (the data_loader casts to category later).
    """
    rng = np.random.default_rng(seed)

    months = _month_starts(start, end)

    # Per-DIN intrinsic attributes.
    dins = [f"{2_000_000 + i:08d}" for i in range(n_dins)]
    drug_codes = [f"DC{1_000_000 + i:07d}" for i in range(n_dins)]
    drug_age_years = rng.uniform(0.0, 40.0, size=n_dins)
    ingredient_count = rng.integers(1, 5, size=n_dins)
    is_pediatric = rng.random(n_dins) < 0.10
    has_atc = rng.random(n_dins) < 0.95
    schedule = rng.choice(SCHEDULES, size=n_dins)
    atc_anatomic = rng.choice(ATC_ANATOMIC, size=n_dins)
    atc_therapeutic = rng.choice(ATC_THERAPEUTIC, size=n_dins)
    primary_route = rng.choice(ROUTES, size=n_dins)
    primary_form = rng.choice(FORMS, size=n_dins)
    formulary_is_generic = rng.random(n_dins) < 0.60

    # Per-DIN "shortage propensity" — the latent each DIN's per-month
    # probability is centred on. Heavy-tailed: a few drugs with much higher
    # rates than the mean, which is what makes ranking interesting.
    propensity = rng.beta(0.5, 12.0, size=n_dins)  # mean ~ 0.04
    propensity = propensity / propensity.mean() * base_shortage_rate

    rows = []
    for di, din in enumerate(dins):
        # Walk through months, sampling shortage starts. We track running
        # counts so the features (shortages_prior_12m, etc.) line up with
        # the actual history.
        prior_starts: list[pd.Timestamp] = []
        currently_short = False
        cur_shortage_end: pd.Timestamp | None = None
        for mi, obs_date in enumerate(months):
            # Resolve current-shortage state.
            if currently_short and cur_shortage_end is not None and obs_date >= cur_shortage_end:
                currently_short = False
                cur_shortage_end = None

            # Lookups against history.
            obs = pd.Timestamp(obs_date)
            t_12m = obs - pd.DateOffset(months=12)
            t_36m = obs - pd.DateOffset(months=36)
            shortages_prior_12m = sum(t_12m <= s < obs for s in prior_starts)
            shortages_prior_36m = sum(t_36m <= s < obs for s in prior_starts)
            shortages_all_prior = sum(s < obs for s in prior_starts)
            if prior_starts:
                last = max(s for s in prior_starts if s < obs) if any(s < obs for s in prior_starts) else None
                first = min(prior_starts)
                days_since_last = (obs - last).days if last is not None else None
                days_since_first = (obs - first).days
            else:
                days_since_last = None
                days_since_first = None

            # Probability of a shortage starting in next 90 days. Driven by
            # propensity + recent history + a small monthly seasonal bump.
            seasonal = 0.005 * np.sin(2 * np.pi * (mi % 12) / 12)
            risk = (
                propensity[di]
                + 0.03 * shortages_prior_12m
                + 0.01 * shortages_prior_36m
                + seasonal
            )
            risk = float(np.clip(risk, 0.001, 0.40))

            # Sample target. If the drug is currently in shortage, the row
            # is excluded from training anyway, so we don't need a fancy
            # joint sampler.
            target = int(rng.random() < risk)

            # If sampled positive, schedule a shortage with a duration drawn
            # from a heavy-tailed distribution.
            if target == 1 and not currently_short:
                duration_days = int(rng.exponential(60.0))
                shortage_start = obs + pd.Timedelta(days=int(rng.integers(0, 90)))
                prior_starts.append(shortage_start)
                # Mark as currently short *next* month if we're inside the window.
                cur_shortage_end = shortage_start + pd.Timedelta(days=duration_days)

            # currently-in-shortage flag (the exclusion flag).
            in_shortage_now = (
                cur_shortage_end is not None
                and obs <= cur_shortage_end
                and currently_short
            )

            # Manufacturer / market features. Sampled with a per-DIN bias so
            # the GBM sees real correlation structure.
            mfr_portfolio_size = int(np.clip(rng.normal(15, 8), 1, 200))
            mfr_short_rate_12m = float(np.clip(propensity[di] * 5 + rng.normal(0, 0.05), 0, 1))
            mfr_short_rate_3m = float(np.clip(mfr_short_rate_12m + rng.normal(0, 0.05), 0, 1))

            row = {
                # Meta
                "observation_date": obs,
                "din": din,
                "drug_code": drug_codes[di],

                # Labels
                TARGET: target,
                EXCLUSION_FLAG: bool(in_shortage_now),

                # Drug intrinsic
                "drug_age_years": float(drug_age_years[di]),
                "ingredient_count": int(ingredient_count[di]),

                # Own-DIN shortage history
                "shortages_prior_12m": shortages_prior_12m,
                "shortages_prior_36m": shortages_prior_36m,
                "shortages_all_prior": shortages_all_prior,
                "days_since_last_shortage": days_since_last,
                "days_since_first_shortage": days_since_first,
                "longest_prior_shortage_days":
                    float(rng.integers(0, 200)) if shortages_all_prior else None,
                "avg_inter_shortage_interval_days":
                    float(rng.integers(30, 800)) if shortages_all_prior > 1 else None,
                "days_overdue":
                    float(rng.integers(0, 60)) if days_since_last and days_since_last > 200 else 0,

                # Manufacturer
                "mfr_portfolio_size": mfr_portfolio_size,
                "mfr_shortages_prior_12m": int(rng.integers(0, 6)),
                "mfr_shortage_rate_12m": mfr_short_rate_12m,
                "mfr_shortage_rate_3m": mfr_short_rate_3m,
                "mfr_shortage_rate_delta_3m_vs_12m": mfr_short_rate_3m - mfr_short_rate_12m,

                # Market structure
                "competing_drugs_same_ai_group": int(rng.integers(1, 30)),
                "mfr_share_of_ai_group": float(np.clip(rng.beta(2, 5), 0, 1)),
                "n_manufacturers_in_ai_group": int(rng.integers(1, 15)),

                # Peer signals
                "peer_shortages_prior_12m_same_ai_group": int(rng.integers(0, 8)),
                "peer_any_in_shortage_now_same_ai_group": bool(rng.random() < 0.20),

                # Discontinuation
                "peer_discontinuations_prior_12m": int(rng.integers(0, 4)),
                "peer_discontinuations_prior_36m": int(rng.integers(0, 10)),
                "days_since_peer_discontinuation":
                    float(rng.integers(0, 1500)) if rng.random() < 0.6 else None,
                "mfr_discontinuations_prior_12m": int(rng.integers(0, 3)),
                "mfr_discontinuation_rate_12m": float(np.clip(rng.beta(1, 30), 0, 1)),

                # Booleans
                "is_pediatric": bool(is_pediatric[di]),
                "has_atc_classification": bool(has_atc[di]),
                "was_ever_in_shortage": shortages_all_prior > 0,
                "formulary_is_generic": bool(formulary_is_generic[di]),

                # Categoricals
                "schedule": schedule[di],
                "atc_anatomic_group": atc_anatomic[di],
                "atc_therapeutic_group": atc_therapeutic[di],
                "primary_route": primary_route[di],
                "primary_form": primary_form[di],

                # FDA features (carried but unused by the unified model)
                "fda_ingredient_match_flag": bool(rng.random() < 0.3),
                "fda_active_ingredients_in_us_shortage": int(rng.integers(0, 3)),
                "fda_ingredients_in_us_shortage_12m": int(rng.integers(0, 6)),
            }

            # Once a shortage has started in the future, mark the DIN as
            # currently-short until the end date passes.
            if target == 1:
                currently_short = True

            rows.append(row)

    df = pd.DataFrame(rows)

    # Sanity: every column declared in config should be present.
    expected = set(META_COLS + [TARGET, EXCLUSION_FLAG]
                   + NUMERIC_FEATURES + BOOLEAN_FEATURES
                   + CATEGORICAL_FEATURES + FDA_FEATURES)
    missing = expected - set(df.columns)
    if missing:
        raise RuntimeError(f"Synthetic panel is missing columns: {sorted(missing)}")

    return df


def build_demo_database(
    db_path: Path | str = DEMO_DB_PATH,
    n_dins: int = 600,
    seed: int = 42,
) -> Path:
    """Populate a DuckDB file with the synthetic panel at PANEL_TABLE.

    Returns the resolved path. Overwrites any existing table at that name.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_synthetic_panel(n_dins=n_dins, seed=seed)
    print(f"Built synthetic panel: {len(df):,} rows, {df['din'].nunique()} DINs, "
          f"{df['observation_date'].nunique()} months, "
          f"target rate {df[TARGET].mean():.4f}")

    schema, table = PANEL_TABLE.split(".")
    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        con.execute(f"DROP TABLE IF EXISTS {PANEL_TABLE}")
        con.register("panel_df", df)
        con.execute(f"CREATE TABLE {PANEL_TABLE} AS SELECT * FROM panel_df")
        n = con.sql(f"SELECT COUNT(*) FROM {PANEL_TABLE}").fetchone()[0]
        print(f"Wrote {n:,} rows to {PANEL_TABLE} in {db_path}")
    finally:
        con.close()
    return db_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db-path", default=str(DEMO_DB_PATH),
                   help=f"Output DuckDB path (default: {DEMO_DB_PATH})")
    p.add_argument("--n-dins", type=int, default=600,
                   help="Number of synthetic DINs (default: 600)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    build_demo_database(db_path=args.db_path, n_dins=args.n_dins, seed=args.seed)


if __name__ == "__main__":
    main()
