SELECT count(*) AS n, count_if(predicted_category IS NULL) AS unparsed FROM preds
