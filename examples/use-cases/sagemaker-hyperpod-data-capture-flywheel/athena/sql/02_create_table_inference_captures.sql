-- Layer 1: the capture envelope. Payloads stay base64 strings; partition projection maps
-- {CAPTURE_LOCATION}YYYY/MM/DD/HH/ to partitions without a crawler.
CREATE EXTERNAL TABLE IF NOT EXISTS inference_captures (
  captureData struct<
    endpointInput:  struct<observedContentType: string, mode: string, data: string, encoding: string>,
    endpointOutput: struct<observedContentType: string, mode: string, data: string, encoding: string>
  >,
  eventMetadata struct<eventId: string, inferenceTime: string>,
  eventVersion string
)
PARTITIONED BY (year string, month string, day string, hour string)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION '${CAPTURE_LOCATION}'
TBLPROPERTIES (
  'projection.enabled'='true',
  'projection.year.type'='integer',  'projection.year.range'='2026,2035',
  'projection.month.type'='integer', 'projection.month.range'='1,12', 'projection.month.digits'='2',
  'projection.day.type'='integer',   'projection.day.range'='1,31',   'projection.day.digits'='2',
  'projection.hour.type'='integer',  'projection.hour.range'='0,23',  'projection.hour.digits'='2',
  'storage.location.template'='${CAPTURE_LOCATION}${year}/${month}/${day}/${hour}/'
)
