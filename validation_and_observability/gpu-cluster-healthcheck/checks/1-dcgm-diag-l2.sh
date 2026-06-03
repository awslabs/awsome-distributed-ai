#!/usr/bin/env bash
# Check 1: DCGM Level 2 Diagnostics
# Runs dcgmi diag -r 2 with JSON output for medium-depth hardware validation.
# Includes deployment readiness, PCIe validation, memory bandwidth, SM stress.
# Runtime: 2.5 - 10.5 minutes depending on GPU count
#
# Severity classification:
#   ISOLATE (Warning level 3) -- Drain node, initiate replacement
#   RESET   (Warning level 2) -- Reboot node, rerun check
#   MONITOR (Warning level 1) -- Keep in service, log for review

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

CHECK_NAME="1-dcgm-diag-l2"
DCGM_TIMEOUT="${DCGM_TIMEOUT:-900}"  # 15-minute default timeout

run_check() {
    init_check "${CHECK_NAME}"

    # Pre-flight: Verify nv-hostengine is running
    log_info "Checking nv-hostengine status"
    if [[ "${DRY_RUN}" != "1" ]]; then
        if ! pgrep -x nv-hostengine > /dev/null 2>&1; then
            log_warn "nv-hostengine is not running -- attempting to start"

            # Check if managed by systemd
            if systemctl_available && systemctl is-active nvidia-dcgm.service > /dev/null 2>&1; then
                log_warn "DCGM is managed by systemd (nvidia-dcgm.service)"
                log_warn "Starting via systemctl to avoid conflicts"
                run_cmd systemctl start nvidia-dcgm.service
            else
                run_cmd nv-hostengine
            fi

            sleep 2
            if ! pgrep -x nv-hostengine > /dev/null 2>&1; then
                check_fail "${CHECK_NAME}" "Unable to start nv-hostengine" "RESET"
                return 1
            fi
        fi
        log_verbose "nv-hostengine is running (PID: $(pgrep -x nv-hostengine))"
    else
        echo -e "${YELLOW}[DRY-RUN]${NC} pgrep nv-hostengine" >&2
    fi

    # Run DCGM Level 2 diagnostics with JSON output
    log_info "Running DCGM Level 2 diagnostics (this may take 2-10 minutes)"
    local dcgm_output
    local dcgm_exit=0

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} timeout ${DCGM_TIMEOUT}s dcgmi diag -r 2 -j" >&2
        check_pass "${CHECK_NAME}" "Dry-run: DCGM L2 diagnostics skipped"
        return 0
    fi

    dcgm_output=$(run_with_timeout "${DCGM_TIMEOUT}" dcgmi diag -r 2 -j 2>&1) || dcgm_exit=$?

    if [[ ${dcgm_exit} -eq 124 ]]; then
        check_fail "${CHECK_NAME}" \
            "DCGM L2 diagnostics timed out after ${DCGM_TIMEOUT}s" "RESET"
        return 1
    fi

    # Save raw output
    echo "${dcgm_output}" > "${RESULTS_DIR}/dcgm-l2-raw.json"
    log_verbose "Raw DCGM output saved to ${RESULTS_DIR}/dcgm-l2-raw.json"

    # Parse results through severity classifier
    local parse_result
    local parse_exit=0
    parse_result=$(echo "${dcgm_output}" | python3 "${SCRIPT_DIR}/../lib/parse-dcgm-results.py" \
        --level 2 2>&1) || parse_exit=$?

    echo "${parse_result}" > "${RESULTS_DIR}/check-${CHECK_NAME}.json"

    if [[ ${parse_exit} -ne 0 ]]; then
        check_fail "${CHECK_NAME}" "Failed to parse DCGM results" "RESET"
        return 1
    fi

    # Extract overall severity from parsed results
    local overall_severity
    overall_severity=$(echo "${parse_result}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('overall_severity', 'UNKNOWN'))
" 2>/dev/null || echo "UNKNOWN")

    local overall_status
    overall_status=$(echo "${parse_result}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('overall_status', 'UNKNOWN'))
" 2>/dev/null || echo "UNKNOWN")

    case "${overall_status}" in
        PASS)
            check_pass "${CHECK_NAME}" "All DCGM L2 tests passed"
            return 0
            ;;
        FAIL)
            check_fail "${CHECK_NAME}" \
                "DCGM L2 diagnostics failed" "${overall_severity}"
            return 1
            ;;
        WARN)
            check_warn "${CHECK_NAME}" \
                "DCGM L2 diagnostics completed with warnings (severity: ${overall_severity})"
            return 0
            ;;
        *)
            check_warn "${CHECK_NAME}" \
                "DCGM L2 diagnostics returned unexpected status: ${overall_status}"
            return 0
            ;;
    esac
}

# ─── Entry point ─────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

run_check
