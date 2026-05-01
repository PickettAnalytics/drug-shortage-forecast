"""Load CIHI Formulary Coverage Data Tool into DuckDB.

Source file: ``formulary-coverage-data-tool-data-table-en.xlsx`` from CIHI's
public Pharmaceutical Data Tool. Sheet "Table 1" carries one row per
(jurisdiction × drug program × DIN × coverage period). The Excel layout is
a merged title row followed by column headers in row 2; we read with
``header=1`` to skip the title.

DINs are stored in the spreadsheet as floats (e.g. ``19.0`` for ``00000019``)
and are zero-padded to 8 characters here so they match the format used
across the rest of the pipeline.

Usage:
    python -m ingest.ingest_cihi
"""

from __future__ import annotations

import duckdb
import pandas as pd

from shortage_forecast.config import RAW_DIR, get_db_path

CIHI_RAW_DIR = RAW_DIR / "CIHI"
EXCEL_FILENAME = "formulary-coverage-data-tool-data-table-en.xlsx"
SHEET_NAME = "Table 1"
RAW_SCHEMA = "raw"
RAW_TABLE = "cihi_formulary_raw"

COLUMN_RENAMES: dict[str, str] = {
    "Jurisdiction":       "jurisdiction",
    "Drug program":       "drug_program",
    "PDIN flag":          "pdin_flag",
    "Brand name":         "brand_name",
    "Active ingredients": "active_ingredients",
    "ATC5 code":          "atc5_code",
    "ATC5 description":   "atc5_description",
    "ATC4 code":          "atc4_code",
    "ATC4 description":   "atc4_description",
    "Drug type":          "drug_type",
    "Benefit status":     "benefit_status",
}

DATE_COLUMNS: list[tuple[str, str]] = [
    ("DIN market date",     "din_market_date"),
    ("Coverage start date", "coverage_start_date"),
    ("Coverage end date",   "coverage_end_date"),
]

KEEP_COLUMNS: list[str] = [
    "din", "jurisdiction", "drug_program", "pdin_flag",
    "brand_name", "active_ingredients",
    "atc5_code", "atc5_description", "atc4_code", "atc4_description",
    "drug_type", "benefit_status",
    "din_market_date", "coverage_start_date", "coverage_end_date",
]


def main() -> None:
    excel_path = CIHI_RAW_DIR / EXCEL_FILENAME
    if not excel_path.exists():
        raise FileNotFoundError(f"CIHI formulary file not found at {excel_path}")

    print(f"Reading {excel_path.name} (sheet '{SHEET_NAME}') ...")
    df = pd.read_excel(excel_path, sheet_name=SHEET_NAME, header=1)
    df.columns = [str(c).strip() for c in df.columns]

    # Drop residual title/junk rows where DIN is non-numeric (the merged
    # title row sometimes bleeds into the first data row).
    df = df[pd.to_numeric(df["DIN"], errors="coerce").notna()].copy()
    print(f"  Raw rows with numeric DIN: {len(df):,}")

    # Zero-pad DIN: Excel stores it as a float (19.0 -> '00000019').
    df["din"] = df["DIN"].apply(lambda x: str(int(float(x))).zfill(8))

    for raw_col, new_col in DATE_COLUMNS:
        df[new_col] = pd.to_datetime(df[raw_col], errors="coerce").dt.date

    df = df.rename(columns=COLUMN_RENAMES)
    df = df[KEEP_COLUMNS]

    print(f"  Final rows: {len(df):,}  |  Unique DINs: {df['din'].nunique():,}")

    db_path = get_db_path()
    print(f"Loading into {db_path.name} as {RAW_SCHEMA}.{RAW_TABLE} ...")
    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA}")
        con.execute(f"DROP TABLE IF EXISTS {RAW_SCHEMA}.{RAW_TABLE}")
        con.register("formulary_df", df)
        con.execute(
            f"CREATE TABLE {RAW_SCHEMA}.{RAW_TABLE} AS SELECT * FROM formulary_df"
        )

        n = con.execute(f"SELECT COUNT(*) FROM {RAW_SCHEMA}.{RAW_TABLE}").fetchone()[0]
        n_dins = con.execute(
            f"SELECT COUNT(DISTINCT din) FROM {RAW_SCHEMA}.{RAW_TABLE}"
        ).fetchone()[0]
        print(f"  Loaded: {n:,} rows, {n_dins:,} unique DINs")
    finally:
        con.close()


if __name__ == "__main__":
    main()
