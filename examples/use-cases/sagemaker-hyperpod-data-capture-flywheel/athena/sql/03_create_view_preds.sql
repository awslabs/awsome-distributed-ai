-- Layer 2: decode base64 payloads once and extract ticket, prediction, confidence.
CREATE OR REPLACE VIEW preds AS
WITH decoded AS (
  SELECT CAST(from_iso8601_timestamp(eventMetadata.inferenceTime) AS timestamp) AS ts,
         from_utf8(from_base64(captureData.endpointInput.data))  AS req,
         from_utf8(from_base64(captureData.endpointOutput.data)) AS resp
  FROM inference_captures
  WHERE year = '${RUN_YEAR}' AND month = '${RUN_MONTH}' AND day = '${RUN_DAY}'
)
SELECT ts,
       json_extract_scalar(req, '$.messages[1].content') AS ticket_text,
       json_extract_scalar(json_extract_scalar(resp, '$.choices[0].message.content'), '$.category') AS predicted_category,
       CAST(json_extract_scalar(json_extract_scalar(resp, '$.choices[0].message.content'), '$.confidence') AS double) AS confidence
FROM decoded
WHERE ts >= timestamp '${RUN_START}' AND ts < timestamp '${RUN_END}'
