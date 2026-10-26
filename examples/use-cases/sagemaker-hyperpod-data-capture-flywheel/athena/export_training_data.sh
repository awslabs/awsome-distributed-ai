#!/usr/bin/env bash
# Export the labeled view through Athena's own CSV output, build chat-format train.jsonl, upload it.
# Usage (from the repo root, after `source env.sh`):  bash athena/export_training_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source athena/run_athena.sh
run_sql athena/sql/13_export_training_rows.sql > /dev/null
aws s3 cp "${ATHENA_OUTPUT}${LAST_QID}.csv" labeled.csv --only-show-errors   # header + proper quoting
python3 scripts/build_training_file.py labeled.csv train.jsonl
aws s3 cp train.jsonl "s3://${BUCKET}/training/train.jsonl"
