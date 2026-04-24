"""
Load FDA drug shortages JSON into DuckDB raw.fda_shortages_raw.

Mirrors the pattern of your existing CSV ingestion: reads from
data/raw/fda/, writes to the raw schema in the DuckDB file that dbt uses.
The table this produces is consumed by stg_fda_shortages.sql.

Why this step exists: dbt-duckdb reads from tables in the DB, not from
files on disk. We need a Python loader to flatten the nested JSON into
a flat table that dbt can read via {{ source('raw', 'fda_shortages_raw') }}.

Usage:
    python load_fda_shortages.py
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


DB_PATH = Path(r"E:\Projects\drug-shortage-forecast\drug_shortages.duckdb")
FDA_DIR = Path(r"E:\Projects\drug-shortage-forecast\data\raw\fda")
RAW_SCHEMA = "raw"
RAW_TABLE = "fda_shortages_raw"


def load_latest_json() -> list[dict]:
    """Read the most recent fda_shortages_*.json file."""
    candidates = sorted(FDA_DIR.glob("fda_shortages_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No fda_shortages_*.json files in {FDA_DIR}")
    latest = candidates[-1]
    print(f"Reading {latest.name}")
    with open(latest, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["results"]


def flatten_record(r: dict) -> dict:
    """
    Flatten one FDA record for the raw table.

    We keep fields that downstream features need:
      - Dates (initial_posting_date, update_date, discontinued_date)
      - Status
      - Active ingredient(s) from openfda.substance_name (exploded separately)
      - Manufacturer, route, dosage form, therapeutic category
      - Free-text reason (related_info) for inspection, not features
    """
    openfda = r.get("openfda") or {}

    def first(field: str) -> str | None:
        values = openfda.get(field)
        if isinstance(values, list) and values:
            return values[0]
        return None

    def joined(field: str) -> str | None:
        values = openfda.get(field)
        if isinstance(values, list) and values:
            return "|".join(str(v) for v in values)
        return None

    therapeutic_cats = r.get("therapeutic_category")
    therapeutic_str = (
        "|".join(therapeutic_cats)
        if isinstance(therapeutic_cats, list) and therapeutic_cats
        else None
    )

    return {
        "initial_posting_date":  r.get("initial_posting_date"),
        "update_date":           r.get("update_date"),
        "discontinued_date":     r.get("discontinued_date"),
        "status":                r.get("status"),
        "update_type":           r.get("update_type"),
        "generic_name":          r.get("generic_name"),
        "company_name":          r.get("company_name"),
        "contact_info":          r.get("contact_info"),
        "related_info":          r.get("related_info"),
        "dosage_form":           r.get("dosage_form"),
        "presentation":          r.get("presentation"),
        "therapeutic_category":  therapeutic_str,
        "openfda_brand_name":        first("brand_name"),
        "openfda_generic_name":      first("generic_name"),
        "openfda_manufacturer_name": first("manufacturer_name"),
        "openfda_route":             first("route"),
        "openfda_product_type":      first("product_type"),
        # substance_name may have multiple entries for combination products;
        # keep them joined for now and let staging explode them.
        "openfda_substance_names_pipe": joined("substance_name"),
    }


def main() -> None:
    records = load_latest_json()
    print(f"Loaded {len(records):,} records from JSON")

    flat = [flatten_record(r) for r in records]
    df = pd.DataFrame(flat)
    print(f"Flattened to DataFrame with shape {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Quick integrity checks
    n_with_substance = df["openfda_substance_names_pipe"].notna().sum()
    n_with_date = df["initial_posting_date"].notna().sum()
    print(f"  Records with substance_name:         {n_with_substance:,} ({n_with_substance/len(df):.1%})")
    print(f"  Records with initial_posting_date:   {n_with_date:,} ({n_with_date/len(df):.1%})")

    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA}")
        con.execute(f"DROP TABLE IF EXISTS {RAW_SCHEMA}.{RAW_TABLE}")
        con.execute(f"CREATE TABLE {RAW_SCHEMA}.{RAW_TABLE} AS SELECT * FROM df")
        row_count = con.sql(f"SELECT COUNT(*) FROM {RAW_SCHEMA}.{RAW_TABLE}").fetchone()[0]
        print(f"\nWrote {row_count:,} rows to {RAW_SCHEMA}.{RAW_TABLE}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
