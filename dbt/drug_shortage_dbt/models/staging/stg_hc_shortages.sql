-- Clean staging model for Health Canada shortage reports.
-- Handles: DIN padding, date casting, column renaming, post-2017 cutoff,
-- boolean coercion.

WITH source AS (
    SELECT * FROM {{ source('raw', 'hc_shortages_raw') }}
),

cleaned AS (
    SELECT
        -- Identifiers
        CAST("Report ID" AS INTEGER) AS report_id,
        CASE 
            WHEN "Drug Identification Number" IS NULL THEN NULL
            ELSE LPAD(
                CAST(CAST("Drug Identification Number" AS BIGINT) AS VARCHAR),
                8,
                '0'
            )
        END AS din,
        ("Drug Identification Number" IS NOT NULL) AS has_din,

        -- Drug attributes
        "Brand name" AS brand_name,
        "Company Name" AS company_name,
        "Common or Proper name" AS common_name,
        "Ingredients" AS ingredients,
        "Strength(s)" AS strength,
        "Packaging size" AS packaging_size,
        "Route of administration" AS route_of_administration,
        "Dosage form(s)" AS dosage_form,

        -- Classification
        "ATC Code" AS atc_code,
        "ATC description" AS atc_description,

        -- Status
        "Report Type" AS report_type,
        "Shortage status" AS shortage_status,
        "Reason" AS reason,
        CASE WHEN "Tier 3" = 'Yes' THEN TRUE ELSE FALSE END AS is_tier_3,

        -- Dates
        TRY_CAST("Anticipated start date" AS DATE) AS anticipated_start_date,
        TRY_CAST("Actual start date" AS DATE)      AS actual_start_date,
        TRY_CAST("Estimated end date" AS DATE)     AS estimated_end_date,
        TRY_CAST("Actual end date" AS DATE)        AS actual_end_date,
        TRY_CAST("Date Created" AS DATE)           AS date_created,
        TRY_CAST("Date Updated" AS DATE)           AS date_updated,

        -- Lineage
        source_year_label,
        source_file

    FROM source
)

SELECT *
FROM cleaned
WHERE actual_start_date >= '2017-03-14'
   OR actual_start_date IS NULL