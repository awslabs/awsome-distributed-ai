SELECT coalesce(corrected_category, '(human review)') AS label, count(*) AS n,
       count(DISTINCT ticket_text) AS distinct_texts,
       count_if(corrected_category <> predicted_category) AS relabeled
FROM labeled_training_data GROUP BY 1 ORDER BY 2 DESC
