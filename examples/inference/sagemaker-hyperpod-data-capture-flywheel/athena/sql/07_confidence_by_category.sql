SELECT predicted_category, count(*) AS n, round(avg(confidence), 3) AS avg_conf,
       approx_percentile(confidence, 0.5) AS median_conf
FROM preds GROUP BY 1 ORDER BY 2 DESC
