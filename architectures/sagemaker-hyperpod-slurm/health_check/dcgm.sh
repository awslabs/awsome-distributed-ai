#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# DCGM diagnostic health check for a single node.
# Stdout contract: HEALTH_CHECK_RESULT:<hostname>:<Passed|Failed|Skipped>:<none|reboot|replace>:<reason>
# "Skipped" = test result is inconclusive (test didn't run or output couldn't be parsed); no remediation applied.
# Severity→remediation ref: https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html#automating-responses-to-dcgm-diagnostic-failures

set -euo pipefail

# --- Constants ---
readonly HOSTNAME=$(hostname)
readonly DCGM_TIMEOUT_L2=1800   # 30m
readonly DCGM_TIMEOUT_L3=3600   # 60m
readonly DCGM_TIMEOUT_L4=10800  # 180m
readonly DCGM_ERROR_NONE=0      # No error
readonly DCGM_ERROR_MONITOR=1   # Informational / monitor-level
readonly DCGM_ERROR_ISOLATE=2   # GPU needs isolation → replace
readonly DCGM_ERROR_UNKNOWN=3   # Unknown severity
readonly DCGM_ERROR_TRIAGE=4    # Needs further investigation
readonly DCGM_ERROR_CONFIG=5    # Software/config issue
readonly DCGM_ERROR_RESET=6     # GPU needs reset → reboot

# --- Helpers ---
die() {
    echo "ERROR: $*" >&2
    echo "HEALTH_CHECK_RESULT:${HOSTNAME}:Skipped:none:$*"
    exit 1
}

emit_fallback_result() { echo "HEALTH_CHECK_RESULT:${HOSTNAME}:Skipped:none:unexpected error (ERR trap)"; exit 1; }
trap emit_fallback_result ERR

# --- Argument parsing ---
DCGM_LEVEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            cat >&2 <<HELPEOF
Usage: $(basename "$0")

DCGM diagnostic health check for a single node.
This script has no CLI options; it is configured via environment variables
set by the orchestrator (health_check_orchestrator.sh).

Environment variables:
  HC_TEST_PARAMS   JSON object with test parameters. Supported keys:
                     "level"  DCGM diagnostic level (2, 3, or 4; default: 4)
                   Example: HC_TEST_PARAMS='{"level": 2}'
  HC_RESULTS_DIR   Directory for output files (default: /tmp)
  HC_TIMESTAMP     Timestamp string for file naming (default: auto-generated)

Output (stdout):
  HEALTH_CHECK_RESULT:<hostname>:<Passed|Failed|Skipped>:<none|reboot|replace>:<reason>
HELPEOF
            exit 0 ;;
        *)         die "Unknown option: $1" ;;
    esac
done

# --- Apply HC_TEST_PARAMS (JSON from orchestrator) for DCGM level ---
if [[ -n "${HC_TEST_PARAMS:-}" ]]; then
    command -v jq &>/dev/null || die "jq is required to parse HC_TEST_PARAMS"
    echo "$HC_TEST_PARAMS" | jq empty 2>/dev/null \
        || die "HC_TEST_PARAMS is not valid JSON: $HC_TEST_PARAMS"

    json_level=$(echo "$HC_TEST_PARAMS" | jq -r '.level // empty')
    if [[ -n "$json_level" ]]; then DCGM_LEVEL="$json_level"; fi
fi

# --- Defaults ---
if [[ -z "$DCGM_LEVEL" ]]; then DCGM_LEVEL=4; fi

OUTPUT_DIR="${HC_RESULTS_DIR:-/tmp}"
FILE_TIMESTAMP="${HC_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

# --- Timeout: always use DCGM level-based defaults ---
# The orchestrator controls the sbatch job wall-clock limit (--time) separately;
# the dcgmi command timeout is always the built-in default for the DCGM level.
TIMEOUT_SECONDS=""
case "$DCGM_LEVEL" in
    2) TIMEOUT_SECONDS=$DCGM_TIMEOUT_L2 ;;
    3) TIMEOUT_SECONDS=$DCGM_TIMEOUT_L3 ;;
    *) TIMEOUT_SECONDS=$DCGM_TIMEOUT_L4 ;;
esac

# --- Validation ---
[[ "$DCGM_LEVEL" =~ ^[2-4]$ ]] || die "DCGM level must be 2-4, got: '$DCGM_LEVEL'"
command -v dcgmi &>/dev/null   || die "dcgmi is not installed or not in PATH"
command -v jq &>/dev/null      || die "jq is not installed or not in PATH"
mkdir -p "$OUTPUT_DIR"         || die "Cannot create output directory: $OUTPUT_DIR"

readonly JSON_FILE="${OUTPUT_DIR}/dcgm_${HOSTNAME}_${FILE_TIMESTAMP}.json"

# --- Run DCGM diagnostic ---
# dcgmi warns when CUDA_VISIBLE_DEVICES is set (Slurm sets it via gres); health check needs all GPUs.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "INFO: Unsetting CUDA_VISIBLE_DEVICES (was '${CUDA_VISIBLE_DEVICES}') — DCGM needs access to all GPUs" >&2
fi
unset CUDA_VISIBLE_DEVICES

echo "=== Running DCGM L${DCGM_LEVEL} diag on ${HOSTNAME} (timeout ${TIMEOUT_SECONDS}s) ===" >&2

dcgmi_rc=0
timeout "${TIMEOUT_SECONDS}" dcgmi diag -r "$DCGM_LEVEL" -j --fail-early > "$JSON_FILE" 2>&1 || dcgmi_rc=$?

# Strip leading non-JSON text (e.g., dcgmi CUDA_VISIBLE_DEVICES warnings) so jq can parse the file.
if [[ -f "$JSON_FILE" && -s "$JSON_FILE" ]]; then
    first_brace=$(grep -n '^{' "$JSON_FILE" | head -1 | cut -d: -f1)
    if [[ -n "$first_brace" && "$first_brace" -gt 1 ]]; then
        echo "WARNING: stripping ${first_brace} leading non-JSON lines from dcgmi output" >&2
        tail -n +"$first_brace" "$JSON_FILE" > "${JSON_FILE}.tmp" && mv "${JSON_FILE}.tmp" "$JSON_FILE"
    fi
fi

if [[ $dcgmi_rc -eq 124 ]]; then
    echo "ERROR: dcgmi timed out after ${TIMEOUT_SECONDS}s" >&2
    trap - ERR
    echo "HEALTH_CHECK_RESULT:${HOSTNAME}:Skipped:none:dcgmi timed out after ${TIMEOUT_SECONDS}s"
    exit 0
elif [[ $dcgmi_rc -ne 0 ]]; then
    echo "WARNING: dcgmi exited with status $dcgmi_rc (may be expected for failures)" >&2
fi

# --- Parse DCGM JSON results ---
# Severity-based remediation: ISOLATE→replace, RESET→reboot.
# All other severities (empty/non-numeric, UNKNOWN, NONE, MONITOR, TRIAGE, CONFIG)
# → no automatic hardware remediation; reported as failures for operator triage.
# "replace" takes precedence over "reboot".
# Outputs two lines: first line is remediation action, second line is reason string.
parse_dcgm_results() {
    local json_file="$1"
    local replace_required=false reboot_required=false
    local failure_reasons=""

    # Infrastructure failures → "error" (Skipped, no remediation)
    [[ -f "$json_file" ]] || { echo "ERROR: output file not found: $json_file" >&2; echo "error"; echo "output file not found"; return; }
    [[ -s "$json_file" ]] || { echo "ERROR: output file is empty: $json_file" >&2; echo "error"; echo "output file is empty"; return; }
    jq empty "$json_file" 2>/dev/null || { echo "ERROR: invalid JSON: $json_file" >&2; echo "error"; echo "invalid JSON output"; return; }

    local categories
    categories=$(jq -r '."DCGM GPU Diagnostic".test_categories[]?.category // empty' "$json_file" 2>/dev/null) || true

    for category in $categories; do
        local tests_json
        tests_json=$(jq -c --arg cat "$category" \
            '."DCGM GPU Diagnostic".test_categories[] | select(.category == $cat) | .tests[]?' \
            "$json_file" 2>/dev/null) || true

        while IFS= read -r test_json; do
            [[ -z "$test_json" ]] && continue
            local test_name
            test_name=$(echo "$test_json" | jq -r '.name // empty') || true
            [[ -z "$test_name" ]] && continue

            local results
            results=$(echo "$test_json" | jq -c '.results[]?' 2>/dev/null) || true

            while IFS= read -r result; do
                [[ -z "$result" ]] && continue
                local status
                status=$(echo "$result" | jq -r '.status // empty') || true
                [[ -z "$status" || "${status,,}" == "pass" ]] && continue

                local gpu_id info severity
                gpu_id=$(echo "$result" | jq -r '.gpu_id // "N/A"') || gpu_id="N/A"
                info=$(echo "$result" | jq -r '.info // empty') || info=""
                severity=$(echo "$result" | jq -r '.error_severity // empty') || severity=""
                echo "FAILURE: ${category}/${test_name} GPU ${gpu_id}: ${info}" >&2

                local failure_msg="${category}/${test_name} GPU ${gpu_id}"
                [[ -n "$info" ]] && failure_msg="${failure_msg} - ${info}"
                if [[ -n "$failure_reasons" ]]; then
                    failure_reasons="${failure_reasons}; ${failure_msg}"
                else
                    failure_reasons="$failure_msg"
                fi

                # ISOLATE → replace (GPU is bad per NVIDIA guidance).
                # RESET   → reboot (GPU needs reset, achieved via node reboot).
                # All other severities (empty/non-numeric, UNKNOWN, NONE, MONITOR,
                # TRIAGE, CONFIG) are reported as failures but left for operator triage.
                if [[ -n "$severity" && "$severity" =~ ^[0-9]+$ ]]; then
                    if (( severity == DCGM_ERROR_ISOLATE )); then
                        replace_required=true
                    elif (( severity == DCGM_ERROR_RESET )); then
                        reboot_required=true
                    fi
                fi
            done <<< "$results"
        done <<< "$tests_json"
    done

    if $replace_required; then echo "replace"
    elif $reboot_required; then echo "reboot"
    else echo "none"; fi

    echo "$failure_reasons"
}

# --- Emit result ---
trap - ERR
if ! parse_output=$(parse_dcgm_results "$JSON_FILE"); then
    echo "ERROR: parse_dcgm_results failed unexpectedly" >&2
    echo "HEALTH_CHECK_RESULT:${HOSTNAME}:Skipped:none:parse_dcgm_results failed unexpectedly"
    exit 0
fi
remediation=$(echo "$parse_output" | head -1)
reason=$(echo "$parse_output" | tail -n +2 | tr '\n' ' ' | sed 's/[[:space:]]*$//')

if [[ "$remediation" == "none" ]]; then
    status="Passed"; reason=""
elif [[ "$remediation" == "error" ]]; then
    status="Skipped"; remediation="none"
else
    status="Failed"
fi

echo "HEALTH_CHECK_RESULT:${HOSTNAME}:${status}:${remediation}:${reason}"
exit 0
