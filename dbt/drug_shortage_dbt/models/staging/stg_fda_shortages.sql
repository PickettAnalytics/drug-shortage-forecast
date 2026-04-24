-- Cleaned staging model for openFDA drug shortage reports.
-- One row per (record × active ingredient) — combination products have
-- multiple ingredients in openfda.substance_names_pipe, and we explode
-- them here so downstream molecule joins are 1:1.
--
-- Temporal discipline: initial_posting_date is the canonical "event date"
-- for leakage filtering. update_date is kept for reference only.

WITH source AS (
    SELECT * FROM {{ source('raw', 'fda_shortages_raw') }}
),

parsed AS (
    SELECT
        -- Dates, parsed from MM/DD/YYYY strings
        TRY_CAST(STRPTIME(initial_posting_date, '%m/%d/%Y') AS DATE) AS initial_posting_date,
        TRY_CAST(STRPTIME(update_date,          '%m/%d/%Y') AS DATE) AS update_date,
        TRY_CAST(STRPTIME(discontinued_date,    '%m/%d/%Y') AS DATE) AS discontinued_date,

        -- Status and lifecycle
        status,
        update_type,

        -- Normalized names (upper/trim for later joining)
        UPPER(TRIM(openfda_substance_names_pipe)) AS substance_names_pipe,
        UPPER(TRIM(COALESCE(openfda_manufacturer_name, company_name))) AS manufacturer_name,
        UPPER(TRIM(openfda_generic_name))          AS generic_name,
        UPPER(TRIM(openfda_brand_name))            AS brand_name,

        -- Descriptors (mostly for narrative / diagnostics, not features)
        UPPER(TRIM(openfda_route))                 AS route,
        UPPER(TRIM(dosage_form))                   AS dosage_form,
        therapeutic_category,
        related_info
    FROM source
),

-- Explode the pipe-delimited substance_names into one row per ingredient.
-- DuckDB's STRING_SPLIT + UNNEST handles the expansion cleanly.
exploded AS (
    SELECT
        initial_posting_date,
        update_date,
        discontinued_date,
        status,
        update_type,
        UPPER(TRIM(CAST(ingredient AS VARCHAR))) AS substance_name,
        manufacturer_name,
        generic_name,
        brand_name,
        route,
        dosage_form,
        therapeutic_category,
        related_info
    FROM parsed,
         UNNEST(STRING_SPLIT(substance_names_pipe, '|')) AS t(ingredient)
    WHERE substance_names_pipe IS NOT NULL
      AND ingredient IS NOT NULL
      AND TRIM(CAST(ingredient AS VARCHAR)) != ''
),

-- Derive analytical flags
final AS (
    SELECT
        *,
        -- "Active shortage" status as of the record's most recent update
        (status = 'Current')              AS is_active_shortage,
        (status = 'Resolved')              AS is_resolved,
        (status = 'To Be Discontinued')    AS is_to_be_discontinued
    FROM exploded
    WHERE initial_posting_date IS NOT NULL
)

SELECT * FROM final
