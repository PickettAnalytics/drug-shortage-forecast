"""Load Health Canada Drug Product Database (DPD) extracts into DuckDB.

DPD is published as four zipped allfile bundles, one per product status:

    allfiles.zip      -> marketed   (drug.txt,    comp.txt,    ...)
    allfiles_ia.zip   -> cancelled  (drug_ia.txt, comp_ia.txt, ...)
    allfiles_ap.zip   -> approved   (drug_ap.txt, comp_ap.txt, ...)
    allfiles_dr.zip   -> dormant    (drug_dr.txt, comp_dr.txt, ...)

Each bundle ships eight entity files (drug, comp, ingred, form, route, ther,
status, schedule). We extract them under ``data/staging/dpd/<category>/``,
load them with the DPD column schema from the Health Canada read-me, drop
the French-translation columns, tag every row with ``product_status_extract``,
and write one ``raw.dpd_<entity>_raw`` table per entity to the DuckDB file
that dbt reads.

Usage:
    python -m ingest.ingest_dpd
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import duckdb
import pandas as pd

from shortage_forecast.config import RAW_DIR, STAGING_DIR, get_db_path

DPD_RAW_DIR = RAW_DIR / "dpd"
DPD_STAGING_DIR = STAGING_DIR / "dpd"
RAW_SCHEMA = "raw"

# Map each ZIP to its product status category and the filename suffix Health
# Canada uses for that status. Per the DPD read-me:
#   marketed  -> no suffix    (drug.txt)
#   cancelled -> _ia          (drug_ia.txt)   [inactivated]
#   approved  -> _ap          (drug_ap.txt)
#   dormant   -> _dr          (drug_dr.txt)
EXTRACTS: dict[str, dict[str, str]] = {
    "allfiles.zip":    {"category": "marketed",  "suffix": ""},
    "allfiles_ia.zip": {"category": "cancelled", "suffix": "_ia"},
    "allfiles_ap.zip": {"category": "approved",  "suffix": "_ap"},
    "allfiles_dr.zip": {"category": "dormant",   "suffix": "_dr"},
}

# Column names from the DPD read-me, in file order. Names ending in _f are
# French translations and are dropped before load.
SCHEMAS: dict[str, list[str]] = {
    "drug": [
        "drug_code", "product_categorization", "class",
        "drug_identification_number", "brand_name", "descriptor",
        "pediatric_flag", "accession_number", "number_of_ais",
        "last_update_date", "ai_group_no",
        "class_f", "brand_name_f", "descriptor_f",
    ],
    "comp": [
        "drug_code", "mfr_code", "company_code", "company_name",
        "company_type", "address_mailing_flag", "address_billing_flag",
        "address_notification_flag", "address_other", "suite_number",
        "street_name", "city_name", "province", "country", "postal_code",
        "post_office_box", "province_f", "country_f",
    ],
    "ingred": [
        "drug_code", "active_ingredient_code", "ingredient",
        "ingredient_supplied_ind", "strength", "strength_unit",
        "strength_type", "dosage_value", "base", "dosage_unit", "notes",
        "ingredient_f", "strength_unit_f", "strength_type_f", "dosage_unit_f",
    ],
    "form": [
        "drug_code", "pharm_form_code",
        "pharmaceutical_form", "pharmaceutical_form_f",
    ],
    "route": [
        "drug_code", "route_of_administration_code",
        "route_of_administration", "route_of_administration_f",
    ],
    "ther": [
        "drug_code", "tc_atc_number", "tc_atc",
        "tc_ahfs_number", "tc_ahfs",
        "tc_atc_f", "tc_ahfs_f",
    ],
    "status": [
        "drug_code", "current_status_flag", "status", "history_date",
        "status_f", "lot_number", "expiration_date",
    ],
    "schedule": [
        "drug_code", "schedule", "schedule_f",
    ],
}


def extract_zip(zip_path: Path, category: str, suffix: str) -> None:
    """Unpack one DPD bundle into ``data/staging/dpd/<category>/``."""
    target_dir = DPD_STAGING_DIR / category
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for entity_name in SCHEMAS:
            member_name = f"{entity_name}{suffix}.txt"
            try:
                with zf.open(member_name) as src, \
                     open(target_dir / member_name, "wb") as dst:
                    dst.write(src.read())
                extracted += 1
            except KeyError:
                # A few entities are missing from some bundles (e.g. some
                # statuses don't ship every file). Skip silently.
                print(f"  (no {member_name} in {zip_path.name})")
    print(f"Extracted {extracted} files from {zip_path.name} -> {category}/")


def load_entity(entity_name: str, columns: list[str]) -> pd.DataFrame:
    """Read every per-category copy of one entity, tag, concat."""
    parts: list[pd.DataFrame] = []
    summary: list[str] = []
    for meta in EXTRACTS.values():
        category = meta["category"]
        suffix = meta["suffix"]
        fpath = DPD_STAGING_DIR / category / f"{entity_name}{suffix}.txt"
        if not fpath.exists():
            summary.append(f"{category}=     0")
            continue
        df = pd.read_csv(
            fpath,
            names=columns,
            encoding="utf-8",
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            low_memory=False,
        )
        # Drop French-translation columns; staging models only consume English.
        french_cols = [c for c in df.columns if c.endswith("_f")]
        df = df.drop(columns=french_cols)
        df["product_status_extract"] = category
        parts.append(df)
        summary.append(f"{category}={len(df):>7,}")
    print(f"{entity_name:<10} {'  '.join(summary)}  total={sum(len(p) for p in parts):>8,}")
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    DPD_STAGING_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: extract every bundle that exists under data/raw/dpd/.
    for zip_filename, meta in EXTRACTS.items():
        zip_path = DPD_RAW_DIR / zip_filename
        if not zip_path.exists():
            print(f"  (skipping {zip_filename}: not present in {DPD_RAW_DIR})")
            continue
        extract_zip(zip_path, meta["category"], meta["suffix"])

    # Step 2: load each entity across categories.
    dataframes: dict[str, pd.DataFrame] = {}
    for entity_name, columns in SCHEMAS.items():
        dataframes[entity_name] = load_entity(entity_name, columns)

    # Step 3: write to DuckDB.
    db_path = get_db_path()
    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA}")
        for entity_name, df in dataframes.items():
            table = f"dpd_{entity_name}_raw"
            con.execute(f"DROP TABLE IF EXISTS {RAW_SCHEMA}.{table}")
            con.register(f"{entity_name}_df", df)
            con.execute(f"CREATE TABLE {RAW_SCHEMA}.{table} AS SELECT * FROM {entity_name}_df")

        print("\n--- Verification ---")
        for entity_name in SCHEMAS:
            table = f"dpd_{entity_name}_raw"
            n = con.execute(f"SELECT COUNT(*) FROM {RAW_SCHEMA}.{table}").fetchone()[0]
            print(f"{RAW_SCHEMA}.{table:<25} total={n:>8,}")

        unique_drugs = con.execute(
            f"SELECT COUNT(DISTINCT drug_code) FROM {RAW_SCHEMA}.dpd_drug_raw"
        ).fetchone()[0]
        unique_dins = con.execute(
            f"SELECT COUNT(DISTINCT drug_identification_number) FROM {RAW_SCHEMA}.dpd_drug_raw "
            "WHERE drug_identification_number IS NOT NULL"
        ).fetchone()[0]
        print(f"\nUnique drug_codes: {unique_drugs:,}")
        print(f"Unique DINs:       {unique_dins:,}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
