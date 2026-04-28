"""Load Health Canada drug shortage / discontinuation CSVs into DuckDB.

Reads zipped yearly exports from data/raw/hc_shortages/, unzips into
data/staging/hc_shortages/ with year-prefixed filenames so they don't
collide, then writes two flat tables to the raw schema in the DuckDB
file that dbt reads.

Usage:
    python -m ingest.ingest_hc
"""

from __future__ import annotations

import zipfile

import duckdb
import pandas as pd

from shortage_forecast.config import RAW_DIR, STAGING_DIR, get_db_path

HC_RAW_DIR = RAW_DIR / "hc_shortages"
HC_STAGING_DIR = STAGING_DIR / "hc_shortages"


def load_and_tag(files: list) -> pd.DataFrame:
    """Read each CSV, tag with its source-year label, concat."""
    dfs = []
    for f in files:
        year_label = f.name.split("__")[0]
        df = pd.read_csv(
            f,
            low_memory=False,
            encoding="utf-8-sig",
            skiprows=1,            # skip the title/disclaimer row
        )
        df["source_year_label"] = year_label
        df["source_file"] = f.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main() -> None:
    HC_STAGING_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Unzip all yearly archives. Each year's files get renamed with the
    # year prefix to avoid collisions. The 2016-2018 file is handled the same
    # way; we'll treat its label as '2016_2018'.
    for zip_path in sorted(HC_RAW_DIR.glob("search_export*.zip")):
        year_label = zip_path.stem.replace("search_export", "")  # e.g. "2019"
        year_label = year_label.replace("-", "_")                # SQL-safe

        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                new_name = f"{year_label}__{member}"
                target = HC_STAGING_DIR / new_name
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
        print(f"Extracted {zip_path.name}")

    # Step 2: Read and concatenate.
    shortage_files = sorted(HC_STAGING_DIR.glob("*shortage_report_export.csv"))
    discontinuation_files = sorted(HC_STAGING_DIR.glob("*discontinuation_report_export.csv"))

    shortages = load_and_tag(shortage_files)
    discontinuations = load_and_tag(discontinuation_files)

    print(f"Shortages: {len(shortages):,} rows across {len(shortage_files)} files")
    print(f"Discontinuations: {len(discontinuations):,} rows across "
          f"{len(discontinuation_files)} files")
    print(f"\nShortage columns: {list(shortages.columns)}")

    # Step 3: Load into DuckDB.
    db_path = get_db_path()
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS raw")
        con.execute("DROP TABLE IF EXISTS raw.hc_shortages_raw")
        con.execute("DROP TABLE IF EXISTS raw.hc_discontinuations_raw")
        con.register("shortages_df", shortages)
        con.register("discontinuations_df", discontinuations)
        con.execute("CREATE TABLE raw.hc_shortages_raw AS SELECT * FROM shortages_df")
        con.execute(
            "CREATE TABLE raw.hc_discontinuations_raw AS SELECT * FROM discontinuations_df"
        )

        print("\n--- Verification ---")
        n_sh = con.execute("SELECT COUNT(*) FROM raw.hc_shortages_raw").fetchone()[0]
        n_dc = con.execute("SELECT COUNT(*) FROM raw.hc_discontinuations_raw").fetchone()[0]
        print(f"{n_sh:,} shortages loaded")
        print(f"{n_dc:,} discontinuations loaded")
    finally:
        con.close()


if __name__ == "__main__":
    main()
