SELECT date_trunc('minute', ts) AS minute, count(*) AS tickets,
       count_if(predicted_category = 'Other') AS other,
       round(100.0 * count_if(predicted_category = 'Other') / count(*), 1) AS other_pct
FROM preds GROUP BY 1 ORDER BY 1
