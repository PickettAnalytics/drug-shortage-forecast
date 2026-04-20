WITH source AS (
    SELECT * FROM {{ source('raw', 'dpd_route_raw') }}
),

cleaned AS (
    SELECT
        CAST(drug_code AS INTEGER) AS drug_code,
        TRY_CAST(route_of_administration_code AS INTEGER) AS route_of_administration_code,
        route_of_administration,
        product_status_extract
    FROM source
)

SELECT * FROM cleaned