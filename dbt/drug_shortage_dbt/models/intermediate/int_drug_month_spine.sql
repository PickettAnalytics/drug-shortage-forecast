-- int_drug_month_spine: one row per (DIN, observation_date) with the label.
-- No features yet — just the observation grid and the target.

WITH months AS (
    SELECT CAST(month_start AS DATE) AS observation_date
    FROM (
        SELECT UNNEST(
            GENERATE_SERIES(
                DATE '2018-01-01',
                DATE '2026-01-01',
                INTERVAL '1 month'
            )
        ) AS month_start
    )
),

drugs AS (
    -- Pre-compute the on-market end date to avoid complex OR conditions
    -- in downstream joins (DuckDB struggles with OR across mixed types).
    SELECT
        din,
        drug_code,
        first_marketed_date,
        CASE 
            WHEN current_status LIKE 'CANCELLED%' THEN current_status_since_date
            ELSE DATE '9999-12-31'
        END AS on_market_until
    FROM {{ ref('dim_drug') }}
    WHERE has_din = TRUE
      AND first_marketed_date IS NOT NULL
),

drug_month_grid AS (
    SELECT
        m.observation_date,
        d.din,
        d.drug_code
    FROM months m
    CROSS JOIN drugs d
    WHERE m.observation_date >= d.first_marketed_date
      AND m.observation_date <= d.on_market_until
),

shortages AS (
    SELECT
        din,
        actual_start_date,
        COALESCE(actual_end_date, DATE '9999-12-31') AS end_date
    FROM {{ ref('fct_shortage_episode') }}
    WHERE din IS NOT NULL
      AND actual_start_date IS NOT NULL
),

ongoing_on_obs AS (
    SELECT
        g.observation_date,
        g.din,
        MAX(1) AS was_in_shortage_on_obs_date
    FROM drug_month_grid g
    INNER JOIN shortages s
        ON g.din = s.din
        AND s.actual_start_date <= g.observation_date
        AND s.end_date > g.observation_date
    GROUP BY g.observation_date, g.din
),

any_shortage_in_window AS (
    SELECT
        g.observation_date,
        g.din,
        MAX(1) AS any_shortage_started_in_window
    FROM drug_month_grid g
    INNER JOIN shortages s
        ON g.din = s.din
        AND s.actual_start_date > g.observation_date
        AND s.actual_start_date <= g.observation_date + INTERVAL '90 days'
    GROUP BY g.observation_date, g.din
),

labeled AS (
    SELECT
        g.observation_date,
        g.din,
        g.drug_code,

        COALESCE(ooo.was_in_shortage_on_obs_date, 0) AS was_in_shortage_on_obs_date,

        CASE
            WHEN COALESCE(aisw.any_shortage_started_in_window, 0) = 1
             AND COALESCE(ooo.was_in_shortage_on_obs_date, 0) = 0
            THEN 1
            ELSE 0
        END AS shortage_started_within_90d

    FROM drug_month_grid g
    LEFT JOIN ongoing_on_obs ooo 
        ON g.observation_date = ooo.observation_date AND g.din = ooo.din
    LEFT JOIN any_shortage_in_window aisw 
        ON g.observation_date = aisw.observation_date AND g.din = aisw.din
)

SELECT * FROM labeled