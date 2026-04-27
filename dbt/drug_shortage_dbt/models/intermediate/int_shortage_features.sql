{{ config(materialized='table') }}

-- int_shortage_features: one row per (din, observation_date), matching the
-- spine, with all predictor features computed as-of the observation date.
--
-- LEAKAGE DISCIPLINE: every feature below is computed using ONLY records
-- strictly prior to observation_date.
--
-- Feature groups:
--   Group 1: Drug intrinsic attributes (static)
--   Group 2: Drug's own shortage history
--   Group 3: Manufacturer portfolio (incl. 3m rate + delta for trajectory)
--   Group 4: Market structure (incl. concentration measures)
--   Group 5: Peer shortage signals (shortages on OTHER DINs in same molecule)
--   Group 6: Discontinuation signals (peer and manufacturer discontinuations)
--   Group 7: Formulary coverage (CIHI) — demand breadth and drug type
--             n_jurisdictions_on_formulary: provinces with active coverage at obs_date (0–12)
--             n_programs_as_benefit: programs listing this DIN as a full Benefit
--             formulary_is_generic / formulary_is_biologics: static DIN classification
--
-- NOTE: An earlier iteration (preserved in git history) tested adding
-- openFDA shortage signals as a seventh group. Subgroup evaluation showed
-- the FDA features did not meaningfully improve the targeted injectable /
-- biologic strata (PR-AUC unchanged on IV, IM, Schedule D), so the FDA
-- features were reverted. The stg_fda_shortages staging model is retained
-- in the repo as evidence of the tested hypothesis.

WITH spine AS (
    SELECT observation_date, din, drug_code
    FROM {{ ref('int_drug_month_spine') }}
),

obs_dates AS (
    SELECT DISTINCT observation_date FROM spine
),

-- === Group 1: Drug attributes (static, no time dimension needed) ===
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

-- Reusable DIN → ai_group_no lookup.
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

-- === Group 2: Drug's own shortage history ===
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

-- === Group 3: Manufacturer portfolio (with trajectory) ===
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
        COUNT(DISTINCT f.report_id) FILTER (
            WHERE f.actual_start_date >= o.observation_date - INTERVAL '12 months'
              AND f.actual_start_date <  o.observation_date
        ) AS mfr_shortages_prior_12m,
        COUNT(DISTINCT f.report_id) FILTER (
            WHERE f.actual_start_date >= o.observation_date - INTERVAL '3 months'
              AND f.actual_start_date <  o.observation_date
        ) AS mfr_shortages_prior_3m
    FROM obs_dates o
    CROSS JOIN drug_attrs_base db
    INNER JOIN {{ ref('fct_shortage_episode') }} f ON db.din = f.din
    WHERE db.company_code IS NOT NULL
      AND f.actual_start_date <  o.observation_date
      AND f.actual_start_date >= o.observation_date - INTERVAL '12 months'
    GROUP BY o.observation_date, db.company_code
),

-- === Group 4: Market structure (with concentration) ===
ai_group_mfr_counts AS (
    SELECT
        o.observation_date,
        ag.ai_group_no,
        dab.company_code,
        COUNT(DISTINCT ag.din) AS n_dins_by_mfr_in_ai_group
    FROM obs_dates o
    CROSS JOIN drug_to_ai_group ag
    INNER JOIN drug_attrs_base dab 
        ON ag.din = dab.din
        AND dab.first_marketed_date <= o.observation_date
    WHERE ag.ai_group_no IS NOT NULL
      AND dab.company_code IS NOT NULL
    GROUP BY o.observation_date, ag.ai_group_no, dab.company_code
),

market_structure_by_date AS (
    SELECT
        observation_date,
        ai_group_no,
        SUM(n_dins_by_mfr_in_ai_group)      AS total_dins_in_ai_group,
        COUNT(DISTINCT company_code)        AS n_manufacturers_in_ai_group
    FROM ai_group_mfr_counts
    GROUP BY observation_date, ai_group_no
),

-- === Group 5: Peer shortage signals ===
ai_group_shortages_by_date AS (
    SELECT
        o.observation_date,
        ag.ai_group_no,
        COUNT(DISTINCT f.report_id) FILTER (
            WHERE f.actual_start_date >= o.observation_date - INTERVAL '12 months'
              AND f.actual_start_date <  o.observation_date
        ) AS ai_group_shortages_prior_12m,
        MAX(CASE
            WHEN f.actual_start_date <  o.observation_date
             AND COALESCE(f.actual_end_date, DATE '9999-12-31') > o.observation_date
            THEN 1 ELSE 0
        END) AS ai_group_any_in_shortage_now
    FROM obs_dates o
    CROSS JOIN drug_to_ai_group ag
    INNER JOIN {{ ref('fct_shortage_episode') }} f ON ag.din = f.din
    WHERE ag.ai_group_no IS NOT NULL
    GROUP BY o.observation_date, ag.ai_group_no
),

din_own_ai_group_contribution AS (
    SELECT
        s.observation_date,
        s.din,
        COUNT(DISTINCT f.report_id) FILTER (
            WHERE f.actual_start_date >= s.observation_date - INTERVAL '12 months'
              AND f.actual_start_date <  s.observation_date
        ) AS own_shortages_prior_12m,
        MAX(CASE
            WHEN f.actual_start_date <  s.observation_date
             AND COALESCE(f.actual_end_date, DATE '9999-12-31') > s.observation_date
            THEN 1 ELSE 0
        END) AS own_in_shortage_now
    FROM spine s
    LEFT JOIN {{ ref('fct_shortage_episode') }} f 
        ON s.din = f.din
    GROUP BY s.observation_date, s.din
),

-- === Group 2b: Shortage cycle timing ===
-- Computes the average interval between consecutive shortage starts per DIN,
-- restricted to shortages strictly prior to observation_date. Requires ≥2
-- prior shortage start dates; NULL otherwise (< 2 prior shortages).
shortage_intervals AS (
    SELECT
        s.observation_date,
        s.din,
        DATEDIFF('day',
            LAG(f.actual_start_date) OVER (
                PARTITION BY s.din, s.observation_date
                ORDER BY f.actual_start_date
            ),
            f.actual_start_date
        ) AS interval_days
    FROM spine s
    INNER JOIN {{ ref('fct_shortage_episode') }} f
        ON s.din = f.din
        AND f.actual_start_date < s.observation_date
    WHERE f.actual_start_date IS NOT NULL
),

shortage_cycle_timing AS (
    SELECT
        observation_date,
        din,
        AVG(interval_days) AS avg_inter_shortage_interval_days
    FROM shortage_intervals
    WHERE interval_days IS NOT NULL  -- NULL for the first shortage (no LAG predecessor)
    GROUP BY observation_date, din
),

-- === Group 7: Formulary coverage (CIHI) ===
--
-- Drug type (Generic/Biologics) is a static property of the DIN in CIHI's
-- database; it does not change across programs or time. Computed once at
-- DIN level using any coverage record, then joined on DIN only.
formulary_din_type AS (
    SELECT
        din,
        MAX(CASE WHEN drug_type = 'Generic'   THEN 1 ELSE 0 END) AS formulary_is_generic,
        MAX(CASE WHEN drug_type = 'Biologics' THEN 1 ELSE 0 END) AS formulary_is_biologics
    FROM {{ ref('stg_cihi_formulary') }}
    GROUP BY din
),

-- Time-scoped coverage at each (din, observation_date):
-- a record is active when coverage_start_date <= observation_date
-- AND (coverage_end_date IS NULL OR coverage_end_date > observation_date).
formulary_coverage AS (
    SELECT
        s.observation_date,
        s.din,
        COUNT(DISTINCT f.jurisdiction) FILTER (
            WHERE f.coverage_start_date <= s.observation_date
              AND (f.coverage_end_date IS NULL OR f.coverage_end_date > s.observation_date)
        ) AS n_jurisdictions_on_formulary,
        COUNT(*) FILTER (
            WHERE f.coverage_start_date <= s.observation_date
              AND (f.coverage_end_date IS NULL OR f.coverage_end_date > s.observation_date)
              AND f.benefit_status = 'Benefit'
        ) AS n_programs_as_benefit
    FROM spine s
    LEFT JOIN {{ ref('stg_cihi_formulary') }} f ON s.din = f.din
    GROUP BY s.observation_date, s.din
),

-- === Group 6: Discontinuation signals ===
discontinuations_enriched AS (
    SELECT
        disc.din                    AS disc_din,
        disc.discontinuation_date,
        ag.ai_group_no              AS disc_ai_group_no,
        dab.company_code            AS disc_company_code
    FROM {{ ref('stg_hc_discontinuations') }} disc
    LEFT JOIN drug_to_ai_group ag   ON disc.din = ag.din
    LEFT JOIN drug_attrs_base dab   ON disc.din = dab.din
    WHERE disc.discontinuation_date IS NOT NULL
      AND disc.has_din = TRUE
),

ai_group_discontinuations_by_date AS (
    SELECT
        o.observation_date,
        de.disc_ai_group_no AS ai_group_no,
        COUNT(*) FILTER (
            WHERE de.discontinuation_date >= o.observation_date - INTERVAL '12 months'
              AND de.discontinuation_date <  o.observation_date
        ) AS ai_group_discontinuations_prior_12m,
        COUNT(*) FILTER (
            WHERE de.discontinuation_date >= o.observation_date - INTERVAL '36 months'
              AND de.discontinuation_date <  o.observation_date
        ) AS ai_group_discontinuations_prior_36m,
        MIN(DATEDIFF('day', de.discontinuation_date, o.observation_date)) FILTER (
            WHERE de.discontinuation_date < o.observation_date
        ) AS ai_group_days_since_last_discontinuation
    FROM obs_dates o
    CROSS JOIN discontinuations_enriched de
    WHERE de.disc_ai_group_no IS NOT NULL
    GROUP BY o.observation_date, de.disc_ai_group_no
),

mfr_discontinuations_by_date AS (
    SELECT
        o.observation_date,
        de.disc_company_code AS company_code,
        COUNT(*) FILTER (
            WHERE de.discontinuation_date >= o.observation_date - INTERVAL '12 months'
              AND de.discontinuation_date <  o.observation_date
        ) AS mfr_discontinuations_prior_12m
    FROM obs_dates o
    CROSS JOIN discontinuations_enriched de
    WHERE de.disc_company_code IS NOT NULL
    GROUP BY o.observation_date, de.disc_company_code
),

-- === Final assembly ===
final AS (
    SELECT
        s.observation_date,
        s.din,
        s.drug_code,

        -- Group 1: Drug attributes
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

        -- Group 2: Drug's own shortage history
        COALESCE(dsh.shortages_prior_12m, 0)    AS shortages_prior_12m,
        COALESCE(dsh.shortages_prior_36m, 0)    AS shortages_prior_36m,
        COALESCE(dsh.shortages_all_prior, 0)    AS shortages_all_prior,
        dsh.days_since_first_shortage,
        dsh.days_since_last_shortage,
        dsh.longest_prior_shortage_days,
        (COALESCE(dsh.shortages_all_prior, 0) > 0) AS was_ever_in_shortage,
        -- Cycle timing (NULL when < 2 prior shortages)
        sct.avg_inter_shortage_interval_days,
        dsh.days_since_last_shortage - sct.avg_inter_shortage_interval_days AS days_overdue,

        -- Group 3: Manufacturer
        COALESCE(mps.mfr_portfolio_size, 0)      AS mfr_portfolio_size,
        COALESCE(msd.mfr_shortages_prior_12m, 0) AS mfr_shortages_prior_12m,
        CASE 
            WHEN mps.mfr_portfolio_size > 0 
            THEN CAST(COALESCE(msd.mfr_shortages_prior_12m, 0) AS DOUBLE) / mps.mfr_portfolio_size
            ELSE 0 
        END AS mfr_shortage_rate_12m,
        CASE
            WHEN mps.mfr_portfolio_size > 0
            THEN CAST(COALESCE(msd.mfr_shortages_prior_3m, 0) AS DOUBLE) / mps.mfr_portfolio_size
            ELSE 0
        END AS mfr_shortage_rate_3m,
        CASE
            WHEN mps.mfr_portfolio_size > 0
            THEN
                (CAST(COALESCE(msd.mfr_shortages_prior_3m, 0) AS DOUBLE) / mps.mfr_portfolio_size)
              - (CAST(COALESCE(msd.mfr_shortages_prior_12m, 0) AS DOUBLE) / mps.mfr_portfolio_size)
            ELSE 0
        END AS mfr_shortage_rate_delta_3m_vs_12m,

        -- Group 4: Market structure
        CASE
            WHEN ms.total_dins_in_ai_group IS NULL THEN 0
            ELSE ms.total_dins_in_ai_group - 1
        END AS competing_drugs_same_ai_group,
        CASE
            WHEN ms.total_dins_in_ai_group > 0
            THEN CAST(COALESCE(agmc.n_dins_by_mfr_in_ai_group, 0) AS DOUBLE) 
                 / ms.total_dins_in_ai_group
            ELSE 0
        END AS mfr_share_of_ai_group,
        COALESCE(ms.n_manufacturers_in_ai_group, 0) AS n_manufacturers_in_ai_group,

        -- Group 5: Peer shortage signals
        GREATEST(
            COALESCE(ags.ai_group_shortages_prior_12m, 0)
              - COALESCE(oc.own_shortages_prior_12m, 0),
            0
        ) AS peer_shortages_prior_12m_same_ai_group,
        CASE
            WHEN COALESCE(ags.ai_group_any_in_shortage_now, 0) = 1
             AND COALESCE(oc.own_in_shortage_now, 0) = 0
            THEN 1
            ELSE 0
        END AS peer_any_in_shortage_now_same_ai_group,

        -- Group 6: Discontinuation signals
        COALESCE(agd.ai_group_discontinuations_prior_12m, 0) 
            AS peer_discontinuations_prior_12m,
        COALESCE(agd.ai_group_discontinuations_prior_36m, 0) 
            AS peer_discontinuations_prior_36m,
        agd.ai_group_days_since_last_discontinuation 
            AS days_since_peer_discontinuation,
        COALESCE(mdd.mfr_discontinuations_prior_12m, 0) 
            AS mfr_discontinuations_prior_12m,
        CASE
            WHEN mps.mfr_portfolio_size > 0
            THEN CAST(COALESCE(mdd.mfr_discontinuations_prior_12m, 0) AS DOUBLE)
                 / mps.mfr_portfolio_size
            ELSE 0
        END AS mfr_discontinuation_rate_12m,

        -- Group 7: Formulary coverage (CIHI)
        COALESCE(fc.n_jurisdictions_on_formulary, 0) AS n_jurisdictions_on_formulary,
        COALESCE(fc.n_programs_as_benefit, 0)        AS n_programs_as_benefit,
        COALESCE(fdt.formulary_is_generic,   0) = 1  AS formulary_is_generic,
        COALESCE(fdt.formulary_is_biologics, 0) = 1  AS formulary_is_biologics

    FROM spine s
    LEFT JOIN drug_attrs_base dab ON s.din = dab.din
    LEFT JOIN drug_shortage_history dsh
        ON s.observation_date = dsh.observation_date AND s.din = dsh.din
    LEFT JOIN shortage_cycle_timing sct
        ON s.observation_date = sct.observation_date AND s.din = sct.din
    LEFT JOIN mfr_portfolio_size_by_date mps 
        ON s.observation_date = mps.observation_date AND dab.company_code = mps.company_code
    LEFT JOIN mfr_shortages_by_date msd 
        ON s.observation_date = msd.observation_date AND dab.company_code = msd.company_code
    LEFT JOIN drug_to_ai_group my_ag ON s.din = my_ag.din
    LEFT JOIN market_structure_by_date ms
        ON s.observation_date = ms.observation_date AND my_ag.ai_group_no = ms.ai_group_no
    LEFT JOIN ai_group_mfr_counts agmc
        ON s.observation_date = agmc.observation_date 
        AND my_ag.ai_group_no = agmc.ai_group_no
        AND dab.company_code  = agmc.company_code
    LEFT JOIN ai_group_shortages_by_date ags
        ON s.observation_date = ags.observation_date AND my_ag.ai_group_no = ags.ai_group_no
    LEFT JOIN din_own_ai_group_contribution oc
        ON s.observation_date = oc.observation_date AND s.din = oc.din
    LEFT JOIN ai_group_discontinuations_by_date agd
        ON s.observation_date = agd.observation_date AND my_ag.ai_group_no = agd.ai_group_no
    LEFT JOIN mfr_discontinuations_by_date mdd
        ON s.observation_date = mdd.observation_date AND dab.company_code = mdd.company_code
    LEFT JOIN formulary_coverage fc
        ON s.observation_date = fc.observation_date AND s.din = fc.din
    LEFT JOIN formulary_din_type fdt
        ON s.din = fdt.din
)

SELECT * FROM final