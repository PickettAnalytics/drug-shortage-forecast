WITH source AS (
    SELECT * FROM {{ source('raw', 'dpd_schedule_raw') }}
),

cleaned AS (
    SELECT
        CAST(drug_code AS INTEGER) AS drug_code,
        schedule,
        product_status_extract
    FROM source
)

SELECT * FROM cleaned