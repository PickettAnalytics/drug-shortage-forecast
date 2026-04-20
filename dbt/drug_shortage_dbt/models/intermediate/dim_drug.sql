-- dim_drug: one row per drug_code (and per DIN, since they're 1:1 in DPD).
-- Collapses the 8 DPD staging tables into a single analytics-ready dimension.
--
-- Design choices:
--   - Multi-ingredient drugs: concatenated into a single string (STRING_AGG)
--   - Multi-form/route drugs: pick one primary + count of alternatives
--   - Status: current status from status flag, plus first-marketed date
--   - Drugs without DINs (radiopharmaceuticals) are included via drug_code

WITH drug_base AS (
    SELECT 
        drug_code,
        din,
        brand_name,
        descriptor,
        is_pediatric,
        number_of_active_ingredients,
        last_update_date AS dpd_last_updated,
        product_status_extract
    FROM {{ ref('stg_dpd_drug') }}
),

company AS (
    SELECT
        drug_code,
        company_code,
        company_name
    FROM {{ ref('stg_dpd_comp') }}
),

ingredients_agg AS (
    -- Combo drugs have multiple rows. Concatenate ingredient names
    -- alphabetically for a stable identifier-like string.
    SELECT
        drug_code,
        STRING_AGG(DISTINCT ingredient, ' + ' ORDER BY ingredient) AS ingredients_list,
        COUNT(DISTINCT ingredient) AS distinct_ingredient_count
    FROM {{ ref('stg_dpd_ingred') }}
    WHERE ingredient IS NOT NULL
    GROUP BY drug_code
),

form_agg AS (
    -- Pick first form alphabetically as primary; count alternatives
    SELECT
        drug_code,
        MIN(pharmaceutical_form) AS primary_form,
        COUNT(DISTINCT pharmaceutical_form) AS distinct_form_count
    FROM {{ ref('stg_dpd_form') }}
    WHERE pharmaceutical_form IS NOT NULL
    GROUP BY drug_code
),

route_agg AS (
    SELECT
        drug_code,
        MIN(route_of_administration) AS primary_route,
        COUNT(DISTINCT route_of_administration) AS distinct_route_count
    FROM {{ ref('stg_dpd_route') }}
    WHERE route_of_administration IS NOT NULL
    GROUP BY drug_code
),

therapeutic AS (
    -- ATC and AHFS are typically 1:1 with drug_code but defensively aggregate
    SELECT
        drug_code,
        MAX(atc_code) AS atc_code,
        MAX(atc_description) AS atc_description,
        MAX(ahfs_code) AS ahfs_code,
        MAX(ahfs_description) AS ahfs_description
    FROM {{ ref('stg_dpd_ther') }}
    GROUP BY drug_code
),

schedule AS (
    SELECT
        drug_code,
        MIN(schedule) AS schedule
    FROM {{ ref('stg_dpd_schedule') }}
    WHERE schedule IS NOT NULL
    GROUP BY drug_code
),

current_status AS (
    -- One row per drug with the current status flag set
    SELECT
        drug_code,
        status AS current_status,
        history_date AS current_status_since_date
    FROM {{ ref('stg_dpd_status') }}
    WHERE is_current_status = TRUE
    -- In rare cases more than one row is flagged current; take the most recent
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY drug_code 
        ORDER BY history_date DESC NULLS LAST
    ) = 1
),

first_marketed AS (
    -- Earliest MARKETED status record per drug
    SELECT
        drug_code,
        MIN(history_date) AS first_marketed_date
    FROM {{ ref('stg_dpd_status') }}
    WHERE status = 'MARKETED'
    GROUP BY drug_code
),

final AS (
    SELECT
        -- Keys
        d.drug_code,
        d.din,

        -- Names & descriptors
        d.brand_name,
        d.descriptor,
        d.is_pediatric,

        -- Ingredients
        i.ingredients_list,
        COALESCE(i.distinct_ingredient_count, 0) AS ingredient_count,
        d.number_of_active_ingredients AS number_of_ais_declared,

        -- Classification
        t.atc_code,
        t.atc_description,
        t.ahfs_code,
        t.ahfs_description,
        (t.atc_code IS NOT NULL) AS has_atc_classification,
        (t.ahfs_code IS NOT NULL) AS has_ahfs_classification,

        -- Pharmaceutical details
        f.primary_form,
        COALESCE(f.distinct_form_count, 0) AS form_count,
        r.primary_route,
        COALESCE(r.distinct_route_count, 0) AS route_count,
        s.schedule,

        -- Company
        c.company_code,
        c.company_name,

        -- Status
        cs.current_status,
        cs.current_status_since_date,
        fm.first_marketed_date,
        d.product_status_extract,

        -- Metadata
        d.dpd_last_updated
    FROM drug_base d
    LEFT JOIN company        c  ON d.drug_code = c.drug_code
    LEFT JOIN ingredients_agg i ON d.drug_code = i.drug_code
    LEFT JOIN form_agg       f  ON d.drug_code = f.drug_code
    LEFT JOIN route_agg      r  ON d.drug_code = r.drug_code
    LEFT JOIN therapeutic    t  ON d.drug_code = t.drug_code
    LEFT JOIN schedule       s  ON d.drug_code = s.drug_code
    LEFT JOIN current_status cs ON d.drug_code = cs.drug_code
    LEFT JOIN first_marketed fm ON d.drug_code = fm.drug_code
)

SELECT * FROM final