-- Clean staging model for Health Canada drug discontinuation reports.
-- Similar structure to shortages but with fewer date columns
-- (discontinuations are point-in-time events, not intervals).

WITH source AS (
    SELECT * FROM {{ source('raw', 'hc_discontinuations_raw') }}
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
        "Discontinuation status" AS discontinuation_status,
        "Reason" AS reason,
        CASE WHEN "Tier 3" = 'Yes' THEN TRUE ELSE FALSE END AS is_tier_3,

        -- Dates (discontinuations only have two date fields, not four)
        TRY_CAST("Discontinuation date" AS DATE)             AS discontinuation_date,
        TRY_CAST("Anticipated discontinuation date" AS DATE) AS anticipated_discontinuation_date,
        TRY_CAST("Date Created" AS DATE)                     AS date_created,
        TRY_CAST("Date Updated" AS DATE)                     AS date_updated,

        -- Lineage
        source_year_label,
        source_file

    FROM source
)

SELECT *
FROM cleaned
WHERE discontinuation_date >= '2017-03-14'
   OR discontinuation_date IS NULL