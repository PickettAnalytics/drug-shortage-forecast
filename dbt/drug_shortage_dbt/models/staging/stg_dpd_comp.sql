WITH source AS (
    SELECT * FROM {{ source('raw', 'dpd_comp_raw') }}
),

cleaned AS (
    SELECT
        CAST(drug_code AS INTEGER) AS drug_code,
        mfr_code,
        TRY_CAST(company_code AS INTEGER) AS company_code,
        company_name,
        company_type,
        UPPER(address_mailing_flag) = 'Y' AS is_mailing_address,
        UPPER(address_billing_flag) = 'Y' AS is_billing_address,
        UPPER(address_notification_flag) = 'Y' AS is_notification_address,
        address_other,
        city_name,
        province,
        country,
        postal_code,
        product_status_extract
    FROM source
)

SELECT * FROM cleaned