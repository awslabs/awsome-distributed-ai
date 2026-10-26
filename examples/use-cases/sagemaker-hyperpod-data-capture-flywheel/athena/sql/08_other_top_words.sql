SELECT word, count(*) AS freq
FROM preds CROSS JOIN UNNEST(regexp_extract_all(lower(ticket_text), '[a-z0-9-]{4,}')) AS t(word)
WHERE predicted_category = 'Other'
  AND word NOT IN ('this','that','with','from','have','since','week','help','need','again','time',
                   'month','second','take','look','someone','order','last','after','when','onto')
GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 10
