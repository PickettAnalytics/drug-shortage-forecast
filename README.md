# Drug Shortage Forecast

A 90-day shortage prediction model for prescription drugs sold in Canada,
built on Health Canada, openFDA, and CIHI public datasets.

The pipeline ingests four public sources, lands them in DuckDB, models a
star-schema panel with dbt, and trains a LightGBM ranker that scores every
drug × month for "will this drug enter shortage in the next 90 days?"

This is a portfolio project. It is not affiliated with Health Canada, the
FDA, or CIHI, and is not intended for clinical or operational use.

## Headline result

Per-month mean Precision@K on a 10-month hold-out (2025-04 .. 2026-01):

| K   | LightGBM + heuristic blend | Heuristic baseline | Lift  |
|----:|---------------------------:|-------------------:|------:|
|  10 |                      0.560 |              0.452 | +24%  |
|  25 |                      0.412 |              0.319 | +29%  |
|  50 |                      0.338 |              0.276 | +22%  |
| 100 |                      0.274 |              0.235 | +17%  |

The base rate of a shortage starting within 90 days for any given drug-month
is roughly 3%. The blended ranker is a within-month rank combination of a
monotone-constrained LightGBM and a `shortages_prior_12m` heuristic — the
two errors are uncorrelated enough that the blend Pareto-dominates either
alone at every K we care about.

See `notebooks/error_analysis.ipynb` for the per-month breakdown and
`notebooks/data_audit.ipynb` for source-data sanity checks.

## Data sources

| Source                     | What it provides                                  | Update cadence |
|----------------------------|---------------------------------------------------|----------------|
| Health Canada Shortages    | Authoritative Canadian shortage / discontinuation | Weekly         |
| Health Canada DPD          | Drug Product Database — DIN, ingredient, ATC, mfr | Daily          |
| openFDA Drug Shortages     | US shortage signals, joinable on active ingredient| Daily          |
| CIHI Formulary Coverage    | Provincial formulary coverage, drug program       | Annual         |

DPD is split into eight tables (drug, comp, ingred, form, route, ther,
status, schedule) — see `dbt/drug_shortage_dbt/models/staging/_sources.yml`
for the full source declaration.

## Architecture

```
                      ┌────────────────────┐
  Public sources ──▶  │  Python ingest     │  ──▶  DuckDB raw schema
                      │  (CSV / JSON / xlsx)│
                      └────────────────────┘
                                │
                                ▼
                      ┌────────────────────┐
                      │  dbt staging        │  cleans, types, deduplicates
                      │  (views)            │
                      └────────────────────┘
                                │
                                ▼
                      ┌────────────────────┐
                      │  dbt intermediate   │  spine, episodes, features
                      │  (views)            │
                      └────────────────────┘
                                │
                                ▼
                      ┌────────────────────┐
                      │  dbt mart           │  mrt_shortage_panel
                      │  (table)            │  one row per DIN × month
                      └────────────────────┘
                                │
                                ▼
                      ┌────────────────────┐
                      │  LightGBM + blend   │  baseline.py + operational_metrics.py
                      └────────────────────┘
```

Feature groups in the panel (49 features in current model):

1. **Drug intrinsic** — age, ingredient count, ATC class, schedule
2. **Own-DIN shortage history** — counts, recency, longest prior episode
3. **Manufacturer signals** — portfolio size, prior-12m shortage rate
4. **Market structure** — competing drugs in same active-ingredient group
5. **Peer signals** — peer shortages / discontinuations in AI group
6. **Discontinuation history** — manufacturer & peer discontinuation rates
7. **CIHI formulary** — generic-vs-brand flag

## Repo layout

```
drug-shortage-forecast/
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example                  # template for API credentials
├── dbt/drug_shortage_dbt/        # dbt project (models, tests)
│   └── models/
│       ├── staging/              # one stg_* view per raw table
│       ├── intermediate/         # spine, episodes, features
│       └── marts/                # mrt_shortage_panel (the modeling table)
└── notebooks/
    ├── data_loader.py            # train/val/test split loader
    ├── baseline.py               # logistic + LightGBM trainers
    ├── operational_metrics.py    # per-month Precision@K + heuristic baselines
    ├── load_fda_shortages.py     # openFDA JSON → DuckDB raw
    ├── data_audit.ipynb          # source-data sanity checks
    └── error_analysis.ipynb      # per-month error & calibration analysis
```

## Setup

Requires Python 3.11+ and ~2 GB free disk for the DuckDB file.

```bash
python -m venv .venv
source .venv/bin/activate          # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env               # then fill in your credentials
```

Configure dbt by adding a profile to `~/.dbt/profiles.yml`:

```yaml
drug_shortage_dbt:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: drug_shortages.duckdb     # relative to where you run dbt
      schema: main
      threads: 4
```

## Running the pipeline

The ingestion scripts that download Health Canada CSVs, the DPD allfiles
extract, and the CIHI formulary spreadsheet are not included in this
snapshot — they live under `data/raw/<source>/` which is gitignored to
keep the repo small. With raw files in place, the rest of the pipeline is:

```bash
# 1. (Optional) Refresh openFDA shortage snapshot
python notebooks/load_fda_shortages.py

# 2. Build dbt models — staging → intermediate → marts
cd dbt/drug_shortage_dbt
dbt deps
dbt build

# 3. Train models, write metrics + plots to ./baseline_results/
cd ../..
python notebooks/baseline.py

# 4. Compare against heuristics with per-month Precision@K
python notebooks/operational_metrics.py
```

## Notes on the model

- **Why ranking, not classification?** The downstream use case is "give me
  the top K drugs to monitor next month," so per-month Precision@K is the
  metric that matches the actual workflow. We report it alongside PR-AUC
  and ROC-AUC for completeness.

- **Why monotone constraints?** Without them, LightGBM learned interaction
  shapes that down-ranked drugs whose shortage history alone would have
  put them at the top — the failure mode where pooled P@10 looked great
  (0.50) but per-month P@10 (0.37) lost to a one-line heuristic. Pinning
  the gradient direction on the most reliable shortage signals fixed this.

- **Why blend with a heuristic?** Log-loss is calibrated across the panel,
  not within month, so the GBM's top-K picks were weak in quiet months
  while pooled top-K were dominated by hot months. Within-month rank
  blending with the `shortages_prior_12m` heuristic eliminates the
  cross-month scale mismatch.

## License

MIT — see [LICENSE](LICENSE).
