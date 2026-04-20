WITH source AS (
    SELECT * FROM {{ source('raw', 'dpd_ingred_raw') }}
),

cleaned AS (
    SELECT
        CAST(drug_code AS INTEGER) AS drug_code,
        TRY_CAST(active_ingredient_code AS INTEGER) AS active_ingredient_code,
        ingredient,
        ingredient_supplied_ind,
        strength,
        strength_unit,
        strength_type,
        dosage_value,
        base,
        dosage_unit,
        notes,
        product_status_extract
    FROM source
)

SELECT * FROM cleaned