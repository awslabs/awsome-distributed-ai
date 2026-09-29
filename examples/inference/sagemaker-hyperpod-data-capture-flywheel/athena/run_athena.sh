# Source this file:  source athena/run_athena.sh
# Then:              run_sql athena/sql/05_other_share_per_minute.sql
#
# Requires: ATHENA_DB, ATHENA_OUTPUT, CAPTURE_LOCATION, RUN_START, RUN_END, LAUNCH_TS (see env.sh.example)

# run_athena "<SQL>": runs one statement, prints tab-separated rows, and sets LAST_QID
run_athena() {
  local qid st
  qid=$(aws athena start-query-execution --query-string "$1" \
        --query-execution-context "Database=$ATHENA_DB" \
        --result-configuration "OutputLocation=$ATHENA_OUTPUT" \
        --output text --query 'QueryExecutionId') || return 1
  while :; do
    st=$(aws athena get-query-execution --query-execution-id "$qid" \
         --output text --query 'QueryExecution.Status.State')
    case "$st" in
      SUCCEEDED) break ;;
      FAILED|CANCELLED) aws athena get-query-execution --query-execution-id "$qid" \
          --output text --query 'QueryExecution.Status.StateChangeReason' >&2; return 1 ;;
    esac
    sleep 2
  done
  LAST_QID=$qid
  aws athena get-query-results --query-execution-id "$qid" \
    --output text --query 'ResultSet.Rows[*].Data[*].VarCharValue'
}

# render_sql <file>: fills in ONLY the variables below, so Athena placeholders such as
# ${year} in the partition-projection template reach Athena untouched.
render_sql() {
  RUN_YEAR="${RUN_START:0:4}" RUN_MONTH="${RUN_START:5:2}" RUN_DAY="${RUN_START:8:2}" \
  python3 - "$1" <<'PY'
import os, re, sys
ALLOWED = {"ATHENA_DB", "CAPTURE_LOCATION", "RUN_START", "RUN_END", "LAUNCH_TS",
           "RUN_YEAR", "RUN_MONTH", "RUN_DAY"}
sql = open(sys.argv[1]).read()
def fill(m):
    name = m.group(1)
    if name not in ALLOWED:
        return m.group(0)
    if not os.environ.get(name):
        sys.exit(f"render_sql: {name} is not set")
    return os.environ[name]
sys.stdout.write(re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", fill, sql))
PY
}

# run_sql <file>: render, then run
run_sql() {
  local sql
  sql=$(render_sql "$1") || return 1
  run_athena "$sql"
}
