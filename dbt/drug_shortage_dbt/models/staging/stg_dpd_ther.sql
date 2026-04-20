WITH source AS (
    SELECT * FROM {{ source('raw', 'dpd_ther_raw') }}
),

cleaned AS (
    SELECT
        CAST(drug_code AS INTEGER) AS drug_code,
        tc_atc_number AS atc_code,
        tc_atc AS atc_description,
        tc_ahfs_number AS ahfs_code,
        tc_ahfs AS ahfs_description,
        product_status_extract
    FROM source
)

SELECT * FROM cleaned