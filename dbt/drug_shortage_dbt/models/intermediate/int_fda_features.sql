-- int_fda_features: FDA shortage signals joined to the DIN observation spine.
-- One row per (din, observation_date), matching int_drug_month_spine grain.
--
-- Join strategy: DPD ingredient names (UPPER/TRIM normalized) are matched
-- against stg_fda_shortages.substance_name (already UPPER/TRIM from staging).
-- Matches are inherently sparse — ingredient naming conventions differ between
-- Canada and the USA — so fda_ingredient_match_flag captures whether any
-- structural match exists at all.
--
-- Temporal discipline: only FDA shortages with initial_posting_date strictly
-- prior to the observation_date are counted (no look-ahead).
--
-- Intended use: cold-start model only (DINs with shortages_all_prior = 0).
-- Features are populated for all spine rows for SQL simplicity; the Python
-- training code restricts them to cold-start rows.

{{ config(materialized='table') }}

WITH

-- Normalize DPD ingredient names per DIN.
-- dim_drug_by_din maps DIN → canonical drug_code (1:1), then join to ingred.
din_ingredients AS (
    SELECT DISTINCT
        dbd.din,
        UPPER(TRIM(i.ingredient)) AS ingredient_normalized
    FROM {{ ref('dim_drug_by_din') }} dbd
    INNER JOIN {{ ref('stg_dpd_ingred') }} i
        ON dbd.drug_code = i.drug_code
    WHERE dbd.din IS NOT NULL
      AND i.ingredient IS NOT NULL
      AND TRIM(i.ingredient) != ''
),

-- Which DINs have at least one ingredient that appears anywhere in the FDA
-- shortage data (time-agnostic)? This is the match-quality flag.
din_fda_ever_matched AS (
    SELECT DISTINCT di.din
    FROM din_ingredients di
    INNER JOIN {{ ref('stg_fda_shortages') }} f
        ON di.ingredient_normalized = f.substance_name
),

-- FDA shortage records we can use for temporal features.
-- substance_name is already UPPER/TRIM from stg_fda_shortages.
fda_timeline AS (
    SELECT
        substance_name,
        initial_posting_date,
        discontinued_date
    FROM {{ ref('stg_fda_shortages') }}
    WHERE initial_posting_date IS NOT NULL
),

spine AS (
    SELECT observation_date, din
    FROM {{ ref('int_drug_month_spine') }}
),

-- Per (din, observation_date): count distinct ingredients in active / recent
-- FDA shortage. Only considers FDA records with initial_posting_date strictly
-- before observation_date (temporal discipline).
--
-- fda_active_ingredients_in_us_shortage:
--   ingredient had an FDA shortage that started before obs_date AND
--   has not been discontinued before obs_date.
--
-- fda_ingredients_in_us_shortage_12m:
--   ingredient had an FDA shortage that started within the 12 months prior
--   to obs_date (active or since resolved — captures recent supply stress).
fda_counts AS (
    SELECT
        s.din,
        s.observation_date,
        COUNT(DISTINCT
            CASE
                WHEN f.initial_posting_date < s.observation_date
                 AND (f.discontinued_date IS NULL OR f.discontinued_date >= s.observation_date)
                THEN di.ingredient_normalized
            END
        ) AS fda_active_ingredients_in_us_shortage,
        COUNT(DISTINCT
            CASE
                WHEN f.initial_posting_date >= s.observation_date - INTERVAL '12 months'
                 AND f.initial_posting_date < s.observation_date
                THEN di.ingredient_normalized
            END
        ) AS fda_ingredients_in_us_shortage_12m
    FROM spine s
    INNER JOIN din_ingredients di ON s.din = di.din
    INNER JOIN fda_timeline f
        ON di.ingredient_normalized = f.substance_name
        AND f.initial_posting_date < s.observation_date  -- temporal gate
    GROUP BY s.din, s.observation_date
)

SELECT
    s.din,
    s.observation_date,

    -- 1 if this DIN's ingredients can be structurally matched to FDA data.
    -- 0 means the ingredient names did not match; FDA counts below will be 0.
    CASE WHEN dfm.din IS NOT NULL THEN 1 ELSE 0 END AS fda_ingredient_match_flag,

    -- Count of DIN's ingredients currently in active shortage in the USA.
    COALESCE(fc.fda_active_ingredients_in_us_shortage, 0) AS fda_active_ingredients_in_us_shortage,

    -- Count of DIN's ingredients that entered a USA shortage in the past 12 months.
    COALESCE(fc.fda_ingredients_in_us_shortage_12m, 0) AS fda_ingredients_in_us_shortage_12m

FROM spine s
LEFT JOIN din_fda_ever_matched dfm ON s.din = dfm.din
LEFT JOIN fda_counts fc ON s.din = fc.din AND s.observation_date = fc.observation_date
