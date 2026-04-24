-- mrt_shortage_panel: the analytics-ready panel for shortage prediction.
-- One row per (din, observation_date). Joins the labeled spine to the
-- feature set and presents them together as the single table the ML
-- pipeline consumes.
--
-- Grain and row count match int_drug_month_spine exactly; the
-- equal_rowcount test guards against silent divergence.

WITH spine AS (
    SELECT * FROM {{ ref('int_drug_month_spine') }}
),

features AS (
    SELECT * FROM {{ ref('int_shortage_features') }}
),

joined AS (
    SELECT
        -- === Keys ===
        s.observation_date,
        s.din,
        s.drug_code,

        -- === Labels ===
        s.shortage_started_within_90d,
        s.was_in_shortage_on_obs_date,

        -- === Drug attributes ===
        f.drug_age_years,
        f.atc_code_full,
        f.atc_anatomic_group,
        f.atc_therapeutic_group,
        f.atc_description,
        f.primary_form,
        f.primary_route,
        f.schedule,
        f.ingredient_count,
        f.is_pediatric,
        f.has_atc_classification,

        -- === Own-drug shortage history ===
        f.shortages_prior_12m,
        f.shortages_prior_36m,
        f.shortages_all_prior,
        f.days_since_last_shortage,
        f.days_since_first_shortage,
        f.longest_prior_shortage_days,
        f.was_ever_in_shortage,

        -- === Manufacturer (with trajectory) ===
        f.mfr_portfolio_size,
        f.mfr_shortages_prior_12m,
        f.mfr_shortage_rate_12m,
        f.mfr_shortage_rate_3m,
        f.mfr_shortage_rate_delta_3m_vs_12m,

        -- === Market structure (with concentration) ===
        f.competing_drugs_same_ai_group,
        f.mfr_share_of_ai_group,
        f.n_manufacturers_in_ai_group,

        -- === Peer shortage signals ===
        f.peer_shortages_prior_12m_same_ai_group,
        f.peer_any_in_shortage_now_same_ai_group,

        -- === Discontinuation signals ===
        f.peer_discontinuations_prior_12m,
        f.peer_discontinuations_prior_36m,
        f.days_since_peer_discontinuation,
        f.mfr_discontinuations_prior_12m,
        f.mfr_discontinuation_rate_12m,

        -- === Metadata ===
        CURRENT_TIMESTAMP AS dbt_loaded_at

    FROM spine s
    LEFT JOIN features f
        ON s.observation_date = f.observation_date
       AND s.din = f.din
)

SELECT * FROM joined