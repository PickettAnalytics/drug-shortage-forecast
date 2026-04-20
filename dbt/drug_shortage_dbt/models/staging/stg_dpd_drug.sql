WITH source AS (
    SELECT * FROM {{ source('raw', 'dpd_drug_raw') }}
),

cleaned AS (
    SELECT
        CAST(drug_code AS INTEGER) AS drug_code,
        product_categorization,
        class,
        drug_identification_number AS din,
        brand_name,
        descriptor,
        UPPER(pediatric_flag) = 'Y' AS is_pediatric,
        accession_number,
        TRY_CAST(number_of_ais AS INTEGER) AS number_of_active_ingredients,
        TRY_CAST(STRPTIME(last_update_date, '%d-%b-%Y') AS DATE) AS last_update_date,
        ai_group_no,
        product_status_extract
    FROM source
)

SELECT * FROM cleaned