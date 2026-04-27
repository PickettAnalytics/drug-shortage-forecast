# drug_shortage_dbt

dbt-DuckDB project for the drug shortage forecast. See the
[project root README](../../README.md) for the full architecture.

## Layers

- `models/staging/` — one `stg_*` view per raw source table; cleans,
  types, deduplicates. Sources are declared in `models/staging/_sources.yml`.
- `models/intermediate/` — `int_drug_month_spine`, `fct_shortage_episode`,
  `dim_drug`, and `int_*_features` views.
- `models/marts/` — `mrt_shortage_panel`, the one-row-per-DIN-per-month
  table the model trains on. Materialized as a table because the loader
  scans it whole.

## Running

Configure `~/.dbt/profiles.yml` (see project root README for the snippet),
then:

```bash
dbt deps         # install dbt_utils
dbt build        # run + test all models
```

Models default to `view` for staging/intermediate and `table` for marts;
override per model with `{{ config(materialized='...') }}`.

## Tests

Schema tests live alongside models in `_schema.yml` files. The tests cover
unique keys on the panel, not-null on the target, and source coverage
(e.g. DPD status date parsing covers ≥99.9% of shortage rows).
