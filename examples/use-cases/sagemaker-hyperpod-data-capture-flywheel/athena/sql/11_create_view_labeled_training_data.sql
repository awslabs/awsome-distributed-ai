-- SQL labeling policy: keyword evidence for the new class, keep other production decisions,
-- mixed signals -> NULL (route to SageMaker Ground Truth).
CREATE OR REPLACE VIEW labeled_training_data AS
WITH t AS (SELECT ticket_text, predicted_category, lower(ticket_text) AS txt FROM preds)
SELECT ticket_text, predicted_category,
  CASE
    -- 1. The new class, identified by keyword evidence, not by what the model predicted
    WHEN regexp_like(txt, '\barctis\b|air condition|\bac unit\b|mini split')
     AND NOT regexp_like(txt, 'fridge|refrigerat|freezer|washer|\boven\b|\brange\b|dishwasher|microwave')
      THEN 'ProductAC'
    -- 2. Keep the production decision when nothing hints at the new product
    WHEN NOT regexp_like(txt, '\barctis\b|air condition|\bac\b|mini split')
      THEN predicted_category
    -- 3. Mixed signals: send to human review (Amazon SageMaker Ground Truth)
    ELSE NULL
  END AS corrected_category
FROM t
