{{ config(materialized='table') }}

-- int_shortage_features: one row per (din, observation_date), matching the
-- spine, with all predictor features computed as-of the observation date.
--
-- LEAKAGE DISCIPLINE: every feature below is computed using ONLY records
-- strictly prior to observation_date.

WITH spine AS (
    SELECT observation_date, din, drug_code
    FROM {{ ref('int_drug_month_spine') }}
),

obs_dates AS (
    SELECT DISTINCT observation_date FROM spine
),

-- === Group 1: Drug attributes (static, no time dimension needed) ===
-- Pulling from dim_drug_by_din guarantees one row per DIN.
drug_attrs_base AS (
    SELECT
        d.din,
        d.drug_code,
        d.first_marketed_date,
        d.atc_code                AS atc_code_full,
        SUBSTRING(d.atc_code, 1, 1) AS atc_anatomic_group,
        SUBSTRING(d.atc_code, 1, 3) AS atc_therapeutic_group,
        d.atc_description,
        d.primary_form,
        d.primary_route,
        d.schedule,
        d.ingredient_count,
        d.is_pediatric,
        d.has_atc_classification,
        d.company_code
    FROM {{ ref('dim_drug_by_din') }} d
),

-- === Group 2: Drug shortage history ===
drug_shortage_history AS (
    SELECT
        s.observation_date,
        s.din,
        COUNT(DISTINCT f.report_id) FILTER (
            WHERE f.actual_start_date >= s.observation_date - INTERVAL '12 months'
              AND f.actual_start_date <  s.observation_date
        ) AS shortages_prior_12m,
        COUNT(DISTINCT f.report_id) FILTER (
            WHERE f.actual_start_date >= s.observation_date - INTERVAL '36 months'
              AND f.actual_start_date <  s.observation_date
        ) AS shortages_prior_36m,
        COUNT(DISTINCT f.report_id) FILTER (
            WHERE f.actual_start_date <  s.observation_date
        ) AS shortages_all_prior,
        MIN(DATEDIFF('day', f.actual_start_date, s.observation_date)) FILTER (
            WHERE f.actual_start_date <  s.observation_date
        ) AS days_since_last_shortage,
        MAX(DATEDIFF('day', f.actual_start_date, s.observation_date)) FILTER (
            WHERE f.actual_start_date <  s.observation_date
        ) AS days_since_first_shortage,
        MAX(f.actual_duration_days) FILTER (
            WHERE f.actual_end_date   <  s.observation_date
              AND f.actual_duration_days IS NOT NULL
        ) AS longest_prior_shortage_days
    FROM spine s
    LEFT JOIN {{ ref('fct_shortage_episode') }} f 
        ON s.din = f.din
        AND f.actual_start_date < s.observation_date
    GROUP BY s.observation_date, s.din
),

-- === Group 3: Manufacturer portfolio ===
mfr_portfolio_size_by_date AS (
    SELECT
        o.observation_date,
        db.company_code,
        COUNT(DISTINCT db.din) AS mfr_portfolio_size
    FROM obs_dates o
    CROSS JOIN drug_attrs_base db
    WHERE db.first_marketed_date <= o.observation_date
      AND db.company_code IS NOT NULL
    GROUP BY o.observation_date, db.company_code
),

mfr_shortages_by_date AS (
    SELECT
        o.observation_date,
        db.company_code,
        COUNT(DISTINCT f.report_id) AS mfr_shortages_prior_12m
    FROM obs_dates o
    CROSS JOIN drug_attrs_base db
    INNER JOIN {{ ref('fct_shortage_episode') }} f ON db.din = f.din
    WHERE db.company_code IS NOT NULL
      AND f.actual_start_date >= o.observation_date - INTERVAL '12 months'
      AND f.actual_start_date <  o.observation_date
    GROUP BY o.observation_date, db.company_code
),

-- === Group 4: Market structure ===
-- Get ai_group_no from stg_dpd_drug, joining via drug_code from dim_drug_by_din.
-- drug_code is already unique per din in dim_drug_by_din, but stg_dpd_drug
-- can have multiple rows per drug_code across status extracts, so dedupe there.
drug_to_ai_group AS (
    SELECT 
        dab.din, 
        dpd.ai_group_no
    FROM drug_attrs_base dab
    LEFT JOIN (
        SELECT drug_code, ai_group_no
        FROM {{ ref('stg_dpd_drug') }}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY drug_code
            ORDER BY 
                CASE product_status_extract
                    WHEN 'marketed'  THEN 1
                    WHEN 'dormant'   THEN 2
                    WHEN 'cancelled' THEN 3
                    WHEN 'approved'  THEN 4
                END
        ) = 1
    ) dpd ON dab.drug_code = dpd.drug_code
),

market_size_by_date AS (
    SELECT
        o.observation_date,
        ag.ai_group_no,
        COUNT(DISTINCT ag.din) AS drugs_in_ai_group
    FROM obs_dates o
    CROSS JOIN drug_to_ai_group ag
    INNER JOIN drug_attrs_base dab 
        ON ag.din = dab.din
        AND dab.first_marketed_date <= o.observation_date
    WHERE ag.ai_group_no IS NOT NULL
    GROUP BY o.observation_date, ag.ai_group_no
),

-- === Final assembly ===
final AS (
    SELECT
        s.observation_date,
        s.din,
        s.drug_code,

        DATEDIFF('day', dab.first_marketed_date, s.observation_date) / 365.25 AS drug_age_years,
        dab.atc_code_full,
        dab.atc_anatomic_group,
        dab.atc_therapeutic_group,
        dab.atc_description,
        dab.primary_form,
        dab.primary_route,
        dab.schedule,
        dab.ingredient_count,
        dab.is_pediatric,
        dab.has_atc_classification,

        COALESCE(dsh.shortages_prior_12m, 0)    AS shortages_prior_12m,
        COALESCE(dsh.shortages_prior_36m, 0)    AS shortages_prior_36m,
        COALESCE(dsh.shortages_all_prior, 0)    AS shortages_all_prior,
        dsh.days_since_first_shortage,
        dsh.days_since_last_shortage,
        dsh.longest_prior_shortage_days,
        (COALESCE(dsh.shortages_all_prior, 0) > 0) AS was_ever_in_shortage,

        COALESCE(mps.mfr_portfolio_size, 0)      AS mfr_portfolio_size,
        COALESCE(msd.mfr_shortages_prior_12m, 0) AS mfr_shortages_prior_12m,
        CASE 
            WHEN mps.mfr_portfolio_size > 0 
            THEN CAST(COALESCE(msd.mfr_shortages_prior_12m, 0) AS DOUBLE) / mps.mfr_portfolio_size
            ELSE 0 
        END AS mfr_shortage_rate_12m,

        CASE
            WHEN msize.drugs_in_ai_group IS NULL THEN 0
            ELSE msize.drugs_in_ai_group - 1
        END AS competing_drugs_same_ai_group

    FROM spine s
    LEFT JOIN drug_attrs_base dab ON s.din = dab.din
    LEFT JOIN drug_shortage_history dsh 
        ON s.observation_date = dsh.observation_date AND s.din = dsh.din
    LEFT JOIN mfr_portfolio_size_by_date mps 
        ON s.observation_date = mps.observation_date AND dab.company_code = mps.company_code
    LEFT JOIN mfr_shortages_by_date msd 
        ON s.observation_date = msd.observation_date AND dab.company_code = msd.company_code
    LEFT JOIN drug_to_ai_group my_ag ON s.din = my_ag.din
    LEFT JOIN market_size_by_date msize 
        ON s.observation_date = msize.observation_date AND my_ag.ai_group_no = msize.ai_group_no
)

SELECT * FROM final