# Drug Shortage Forecast — pipeline entrypoints.
#
# Standard targets:
#   make ingest        Load raw sources into DuckDB (Health Canada + openFDA)
#   make transform     Run dbt models (staging -> intermediate -> marts)
#   make test          Run dbt tests
#   make train         Train baseline models and write metrics to ./baseline_results/
#   make full_pipeline ingest -> transform -> test -> train, end to end
#
# Assumes Python and dbt are on PATH and a profile named drug_shortage_dbt
# is configured per the README. Override PYTHON= or DBT= on the command line
# if you need to point at a different interpreter, e.g.:
#   make train PYTHON=.venv/Scripts/python

PYTHON  ?= python
DBT     ?= dbt
DBT_DIR := dbt/drug_shortage_dbt

.PHONY: help ingest transform test train full_pipeline

help:
	@echo "Targets:"
	@echo "  ingest         Load raw sources into DuckDB"
	@echo "  transform      Run dbt models"
	@echo "  test           Run dbt tests"
	@echo "  train          Train baseline models and write metrics"
	@echo "  full_pipeline  Run ingest -> transform -> test -> train"

ingest:
	$(PYTHON) src/ingest/ingest_hc.py
	$(PYTHON) notebooks/load_fda_shortages.py

transform:
	cd $(DBT_DIR) && $(DBT) deps && $(DBT) run

test:
	cd $(DBT_DIR) && $(DBT) test

train:
	$(PYTHON) notebooks/baseline.py

full_pipeline: ingest transform test train
