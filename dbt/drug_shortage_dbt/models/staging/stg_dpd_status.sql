WITH source AS (
    SELECT * FROM {{ source('raw', 'dpd_status_raw') }}
),

cleaned AS (
    SELECT
        CAST(drug_code AS INTEGER) AS drug_code,
        UPPER(current_status_flag) = 'Y' AS is_current_status,
        status,
        TRY_CAST(history_date AS DATE) AS history_date,
        lot_number,
        TRY_CAST(expiration_date AS DATE) AS expiration_date,
        product_status_extract
    FROM source
)

SELECT * FROM cleaned