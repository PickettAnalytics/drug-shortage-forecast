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
├── pyproject.toml                # package metadata, ruff + pytest config
├── requirements.txt              # pinned runtime deps
├── Makefile                      # `make demo`, `make train`, `make pytest`, ...
├── .env.example                  # template for API credentials
├── .github/
│   ├── workflows/ci.yml          # ruff + pytest + dbt parse on push/PR
│   └── dbt/profiles.yml          # CI-only dbt profile
├── dbt/drug_shortage_dbt/        # dbt project (models, tests)
│   └── models/
│       ├── staging/              # one stg_* view per raw table
│       ├── intermediate/         # spine, episodes, features
│       └── marts/                # mrt_shortage_panel (the modeling table)
├── src/
│   ├── shortage_forecast/        # modelling package
│   │   ├── config.py             # paths, splits, feature groups, hyperparams
│   │   ├── data_loader.py        # train/val/test split loader
│   │   ├── baseline.py           # logistic + LightGBM trainers
│   │   ├── operational.py        # per-month Precision@K + heuristic baselines
│   │   └── demo.py               # synthetic-panel builder for offline runs
│   └── ingest/                   # raw-source loaders
│       ├── ingest_hc.py          # Health Canada CSV → DuckDB raw
│       └── load_fda_shortages.py # openFDA JSON → DuckDB raw
├── tests/                        # pytest suite (config, metrics, models, demo)
└── notebooks/
    ├── data_audit.ipynb          # source-data sanity checks
    ├── error_analysis.ipynb      # per-month error & calibration analysis
    └── model_summary.ipynb       # write-up notebook
```

## Setup

Requires Python 3.11+ and ~2 GB free disk for the DuckDB file.

```bash
python -m venv .venv
source .venv/bin/activate          # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"            # editable install + pytest/ruff
cp .env.example .env               # then fill in your credentials
```

`pip install -e .` registers the `shortage_forecast` and `ingest`
packages on the PYTHONPATH. If you'd rather not install, the `Makefile`
prepends `src/` to `PYTHONPATH` for every Python entry point so the
targets work either way.

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

## Running the Pipeline

The pipeline is wrapped in a `Makefile` at the repo root. The end-to-end
run is one line:

```bash
make full_pipeline
```

That expands to four steps you can also run individually:

```bash
make ingest      # load raw HC + openFDA into DuckDB raw schema
make transform   # dbt deps + dbt run (staging -> intermediate -> marts)
make dbt-test    # dbt test
make train       # baseline trainer — writes ./baseline_results/
```

For a heuristic-vs-model comparison with per-month Precision@K:

```bash
make operational
```

If you don't have GNU `make` on your platform (common on Windows without
WSL), each target maps to a one-liner you can copy out of the `Makefile`,
or invoke the modules directly:

```bash
python -m shortage_forecast.baseline      # train + write metrics
python -m shortage_forecast.operational   # per-month operational metrics
python -m ingest.ingest_hc                # ingest Health Canada CSVs
```

### Tests, lint, CI

```bash
make pytest      # Python unit tests (no DuckDB needed; demo built on the fly)
make lint        # ruff check on src/ and tests/
make dbt-test    # dbt schema tests (requires populated DuckDB)
```

The same three checks run on every push and pull request — see
`.github/workflows/ci.yml`. CI builds a small synthetic DuckDB before
running `dbt parse` so it can validate the project without real data.

## Reproducibility & Data Access

The raw datasets that feed this pipeline — Health Canada shortage exports,
the DPD allfiles extract, openFDA's drug-shortages JSON, and CIHI's
formulary coverage spreadsheet — are **not committed to this repository**.
Health Canada and CIHI publish under terms that distinguish "use" from
"redistribution," and bundling their files into a portfolio repo would
cross that line. The files are also large enough that committing them
would balloon clone time for no real benefit.

### Expected inputs

The pipeline expects raw files under `data/raw/`, organised by source:

```
data/raw/
├── hc_shortages/        # search_export*.zip from drugshortagescanada.ca
├── dpd/                 # DPD allfiles extract (drug, ingred, form, ...)
├── fda/                 # fda_shortages_*.json from openFDA
├── CIHI/                # provincial formulary coverage spreadsheet
└── recalls/             # Health Canada recall feed (optional)
```

The dbt mart `mrt_shortage_panel` is the canonical modelling table — one
row per `(observation_date, din)` for monthly observation dates from
2018-01 to the most recent month with a complete 90-day forward label.
Each row carries the binary `shortage_started_within_90d` target and the
~50 features documented under "Feature groups" above. Source column-level
declarations live in `dbt/drug_shortage_dbt/models/staging/_sources.yml`.

### What can be run without raw data

Even with empty `data/raw/`, you can:

- **Run the full modelling pipeline against a synthetic panel**:
  ```bash
  make demo-train         # build synthetic DuckDB, train baseline against it
  make demo-operational   # run heuristic + blend comparison on the synthetic DB
  ```
  These targets call `python -m shortage_forecast.demo` to generate
  `drug_shortages_demo.duckdb` (one synthetic row per DIN per month, with
  every feature column populated), then point the trainer at it via the
  `DRUG_SHORTAGE_DB` environment variable. Numbers are not interpretable —
  the data is random — but the full code path runs end-to-end so you can
  inspect the metrics tables and the artefacts under `demo_results/`.
- **Run the test suite**:
  ```bash
  make pytest
  ```
  The fixtures build their own synthetic DuckDB; no raw data needed.
- **Read the dbt project** — model SQL, schema tests, and the panel
  definition are all checked in under `dbt/drug_shortage_dbt/`. `dbt parse`
  and `dbt compile` run without the database populated.
- **Read the modelling code** under `src/shortage_forecast/`.
- **Inspect the EDA / model summary notebooks**; their saved cell outputs
  include the headline tables and figures.

`make transform`, `make dbt-test`, and `make train` all require a
populated DuckDB file at `drug_shortages.duckdb`. `make demo-train` and
`make pytest` do not.

### Plugging in your own data

If you have monthly drug-level shortage data with at least
`(observation_date, drug_id, shortage_started_within_90d)`, you can adapt
this project by:

1. Replacing the staging models in `dbt/drug_shortage_dbt/models/staging/`
   so they point at your sources, while keeping the column names the
   intermediate models consume.
2. Updating `SplitConfig` in `src/shortage_forecast/config.py` so the
   train / val / test windows match your data range.
3. Running `make full_pipeline`.

## Notes on the model

- **Why ranking, not classification?** The downstream use case is "give me
  the top K drugs to monitor next month," so per-month Precision@K is the
  metric that matches the actual workflow. We report it alongside PR-AUC
  and ROC-AUC for completeness.

- **Why monotone constraints?** Without them, LightGBM learned interaction
  shapes that down-ranked drugs whose shortage history alone would have
  put them at the top. Pinning the gradient direction on the most reliable
  shortage signals improved this.

- **Why blend with a heuristic?** Log-loss is calibrated across the panel,
  not within month, so the GBM's top-K picks were weak in quiet months
  while pooled top-K were dominated by hot months. Within-month rank
  blending with the `shortages_prior_12m` heuristic eliminates the
  cross-month scale mismatch.

## License

MIT — see [LICENSE](LICENSE).
