SELECT IF(ts < timestamp '${LAUNCH_TS}', '1-before launch', '2-after launch') AS period,
       predicted_category, count(*) AS n
FROM preds GROUP BY 1, 2 ORDER BY 1, 3 DESC
