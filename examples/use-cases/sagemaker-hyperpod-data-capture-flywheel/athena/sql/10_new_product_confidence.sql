SELECT predicted_category, confidence, count(*) AS n FROM preds
WHERE regexp_like(lower(ticket_text), 'arctis|air condition')
GROUP BY 1, 2 ORDER BY 1, 2 DESC
