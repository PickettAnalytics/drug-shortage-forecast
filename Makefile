# Drug Shortage Forecast — pipeline entrypoints.
#
# Standard targets:
#   make ingest         Load raw sources into DuckDB (Health Canada + openFDA)
#   make transform      Run dbt models (staging -> intermediate -> marts)
#   make dbt-test       Run dbt tests
#   make train          Train baseline models, write metrics to ./baseline_results/
#   make operational    Run per-month operational metrics (model + heuristics)
#   make full_pipeline  ingest -> transform -> dbt-test -> train, end to end
#
# Demo / dev targets (no real raw data required):
#   make demo           Build a synthetic DuckDB at ./drug_shortages_demo.duckdb
#   make demo-train     Train baseline against the synthetic DB
#   make demo-operational  Operational metrics against the synthetic DB
#   make lint           Ruff lint on src/ + tests/
#   make pytest         Run the Python test suite
#
# Override PYTHON= or DBT= on the command line if needed, e.g.:
#   make train PYTHON=.venv/Scripts/python

PYTHON  ?= python
DBT     ?= dbt
DBT_DIR := dbt/drug_shortage_dbt

# All Python entry points run with src/ on PYTHONPATH so the package is
# importable without `pip install -e .`. Anyone who has installed the
# package via pip can ignore this prefix; it's a no-op in that case.
RUN_PY := PYTHONPATH=src $(PYTHON)

DEMO_DB := drug_shortages_demo.duckdb

.PHONY: help ingest transform dbt-test train operational full_pipeline \
        demo demo-train demo-operational lint pytest test

help:
	@echo "Targets:"
	@echo "  ingest             Load raw sources into DuckDB"
	@echo "  transform          Run dbt models"
	@echo "  dbt-test           Run dbt tests"
	@echo "  train              Train baseline models and write metrics"
	@echo "  operational        Per-month operational metrics + heuristic baselines"
	@echo "  full_pipeline      ingest -> transform -> dbt-test -> train"
	@echo ""
	@echo "Demo / dev (no raw data required):"
	@echo "  demo               Build synthetic DuckDB at $(DEMO_DB)"
	@echo "  demo-train         Train against the synthetic DB"
	@echo "  demo-operational   Run operational metrics against the synthetic DB"
	@echo "  lint               Ruff lint on src/ + tests/"
	@echo "  pytest             Run the Python test suite"

# --- Real pipeline ---------------------------------------------------------

ingest:
	$(RUN_PY) -m ingest.ingest_hc
	$(RUN_PY) -m ingest.load_fda_shortages

transform:
	cd $(DBT_DIR) && $(DBT) deps && $(DBT) run

dbt-test:
	cd $(DBT_DIR) && $(DBT) test

train:
	$(RUN_PY) -m shortage_forecast.baseline

operational:
	$(RUN_PY) -m shortage_forecast.operational

full_pipeline: ingest transform dbt-test train

# --- Demo / dev ------------------------------------------------------------

demo:
	$(RUN_PY) -m shortage_forecast.demo --db-path $(DEMO_DB)

demo-train: demo
	DRUG_SHORTAGE_DB=$(DEMO_DB) DRUG_SHORTAGE_OUTPUT_DIR=demo_results \
	  $(RUN_PY) -m shortage_forecast.baseline

demo-operational: demo
	DRUG_SHORTAGE_DB=$(DEMO_DB) $(RUN_PY) -m shortage_forecast.operational

lint:
	ruff check src tests

pytest:
	$(RUN_PY) -m pytest

test: pytest dbt-test
