WITH shortages AS (
    SELECT * FROM {{ ref('stg_hc_shortages') }}
),

drug_dim AS (
    -- Dedupe the rare DIN that appears under multiple drug_codes:
    -- prefer marketed > dormant > cancelled > approved, then lowest drug_code.
    SELECT *
    FROM {{ ref('dim_drug') }}
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY din
        ORDER BY 
            CASE product_status_extract
                WHEN 'marketed'  THEN 1
                WHEN 'dormant'   THEN 2
                WHEN 'cancelled' THEN 3
                WHEN 'approved'  THEN 4
            END,
            drug_code
    ) = 1
),

joined AS (
    SELECT
        s.report_id,
        s.din,
        s.has_din,

        s.brand_name                 AS reported_brand_name,
        s.company_name               AS reported_company_name,
        s.atc_code                   AS reported_atc_code,
        s.atc_description            AS reported_atc_description,
        s.reason,
        s.shortage_status,
        s.is_tier_3,

        s.anticipated_start_date,
        s.actual_start_date,
        s.estimated_end_date,
        s.actual_end_date,
        s.date_created,
        s.date_updated,

        -- Durations & lead times
        --
        -- Avoided shortages: actual_start_date is the anticipated future date,
        -- so any calculated duration would be misleading.
        --
        -- ~6 resolved records from 2017-2018 have end_date < start_date,
        -- likely data entry errors during the first year of mandatory reporting.
        CASE 
            WHEN s.shortage_status = 'Avoided shortage'       THEN NULL
            WHEN s.actual_end_date < s.actual_start_date      THEN NULL
            ELSE DATEDIFF('day', s.actual_start_date, s.actual_end_date)
        END AS actual_duration_days,
        DATEDIFF('day', s.anticipated_start_date, s.actual_start_date) AS lead_time_days,
        DATEDIFF('day', s.estimated_end_date, s.actual_end_date) AS duration_estimate_error_days,
        (s.shortage_status != 'Avoided shortage' 
         AND s.actual_end_date < s.actual_start_date) AS has_suspect_dates,
        DATEDIFF('day', s.anticipated_start_date, s.actual_start_date) AS lead_time_days,
        DATEDIFF('day', s.estimated_end_date, s.actual_end_date) AS duration_estimate_error_days,

        (s.shortage_status = 'Resolved')             AS is_resolved,
        (s.shortage_status = 'Actual shortage')      AS is_active,
        (s.shortage_status = 'Anticipated shortage') AS is_anticipated,
        (s.shortage_status = 'Avoided shortage')     AS is_avoided,

        d.drug_code                   AS dpd_drug_code,
        d.brand_name                  AS dpd_brand_name,
        d.ingredients_list            AS dpd_ingredients_list,
        d.ingredient_count            AS dpd_ingredient_count,
        d.atc_code                    AS dpd_atc_code,
        d.atc_description             AS dpd_atc_description,
        d.ahfs_code                   AS dpd_ahfs_code,
        d.ahfs_description            AS dpd_ahfs_description,
        d.primary_form                AS dpd_primary_form,
        d.primary_route               AS dpd_primary_route,
        d.schedule                    AS dpd_schedule,
        d.company_code                AS dpd_company_code,
        d.company_name                AS dpd_company_name,
        d.current_status              AS dpd_current_status,
        d.first_marketed_date         AS dpd_first_marketed_date,
        d.product_status_extract      AS dpd_product_status_extract,

        (d.drug_code IS NOT NULL)     AS matched_to_dpd,

        s.source_year_label,
        s.source_file

    FROM shortages s
    LEFT JOIN drug_dim d ON s.din = d.din
)

SELECT * FROM joined