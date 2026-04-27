-- Cleaned staging model for CIHI formulary coverage data.
-- One row per (jurisdiction × drug_program × DIN × coverage_period).
-- DIN is already zero-padded to 8 characters by the ingest script.
--
-- Source: CIHI Formulary Coverage Data Tool (Table 1)
-- Covers 12 Canadian jurisdictions, 1989–present.
-- coverage_end_date IS NULL means the coverage is still active.

WITH source AS (
    SELECT * FROM {{ source('raw', 'cihi_formulary_raw') }}
),

cleaned AS (
    SELECT
        din,
        TRIM(jurisdiction)       AS jurisdiction,
        TRIM(drug_program)       AS drug_program,
        TRIM(pdin_flag)          AS pdin_flag,
        TRIM(brand_name)         AS brand_name,
        TRIM(active_ingredients) AS active_ingredients,
        TRIM(atc5_code)          AS atc5_code,
        TRIM(atc5_description)   AS atc5_description,
        TRIM(atc4_code)          AS atc4_code,
        TRIM(atc4_description)   AS atc4_description,
        TRIM(drug_type)          AS drug_type,
        TRIM(benefit_status)     AS benefit_status,
        TRY_CAST(din_market_date     AS DATE) AS din_market_date,
        TRY_CAST(coverage_start_date AS DATE) AS coverage_start_date,
        TRY_CAST(coverage_end_date   AS DATE) AS coverage_end_date
    FROM source
    WHERE din IS NOT NULL
      AND coverage_start_date IS NOT NULL
)

SELECT * FROM cleaned
