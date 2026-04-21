-- dim_drug_by_din: one row per DIN, picking the best representative
-- drug_code when multiple exist.
--
-- Some DINs appear in DPD under multiple drug_codes (e.g., same product
-- registered under both marketed and cancelled extracts). This model gives
-- us a single canonical row per DIN, so any downstream model that joins
-- on DIN doesn't have to worry about row multiplication.
--
-- Preference order when a DIN has multiple drug_codes:
--   marketed > dormant > cancelled > approved, then lowest drug_code.

SELECT *
FROM {{ ref('dim_drug') }}
WHERE has_din = TRUE
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