WITH source AS (
    SELECT * FROM {{ source('raw', 'dpd_form_raw') }}
),

cleaned AS (
    SELECT
        CAST(drug_code AS INTEGER) AS drug_code,
        TRY_CAST(pharm_form_code AS INTEGER) AS pharm_form_code,
        pharmaceutical_form,
        product_status_extract
    FROM source
)

SELECT * FROM cleaned