#!/usr/bin/env bash
# Check 4: DCGM Level 4 Diagnostics (Quarantine / Post-Mortem Only)
# Runs dcgmi diag -r 4 with comprehensive pre-flight validation.
# Includes EUD (~20 min), pulse power test, long-duration memory/SM stress.
# Runtime: 45 minutes to 2.25 hours depending on GPU count
#
# CRITICAL: This check requires exclusive node access. Do NOT run on
# nodes with active workloads or concurrent GPU telemetry.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

CHECK_NAME="4-dcgm-diag-l4"
DCGM_L4_TIMEOUT="${DCGM_L4_TIMEOUT:-9000}"  # 2.5 hours default timeout

# Services that hold GPU resources and may interfere with L4 diagnostics.
# Note: nv-fabricmanager manages NVSwitch fabric and is NOT telemetry --
# stopping it can break GPU connectivity needed for diagnostics.
GPU_TELEMETRY_SERVICES=(
    "dcgm-exporter"
    "nvidia-dcgm-exporter"
)

STOPPED_SERVICES=()

restore_services() {
    if [[ ${#STOPPED_SERVICES[@]} -gt 0 ]] && systemctl_available; then
        log_info "Restoring previously stopped services"
        for svc in "${STOPPED_SERVICES[@]}"; do
            log_info "Restarting ${svc}.service"
            systemctl start "${svc}.service" 2>/dev/null || \
                log_warn "Failed to restart ${svc}.service -- manual restart may be needed"
        done
    fi
}

# Ensure services are restored on any exit (SIGTERM, set -e, etc.)
trap restore_services EXIT

run_check() {
    init_check "${CHECK_NAME}"

    echo ""
    log_warn "╔══════════════════════════════════════════════════════════════╗"
    log_warn "║  DCGM Level 4 -- Deep Diagnostic (Quarantine Mode)          ║"
    log_warn "║                                                            ║"
    log_warn "║  • Runtime: 45 min – 2.25 hr                               ║"
    log_warn "║  • Requires exclusive node access                          ║"
    log_warn "║  • Pulse test causes variable power draw                   ║"
    log_warn "║  • EUD cannot run with MIG enabled                         ║"
    log_warn "║  • No concurrent GPU telemetry during test                 ║"
    log_warn "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} Pre-flight checks and dcgmi diag -r 4 -j skipped" >&2
        check_pass "${CHECK_NAME}" "Dry-run: DCGM L4 diagnostics skipped"
        return 0
    fi

    # ── Pre-flight 1: Verify exclusive allocation ────────────────────────
    log_info "Pre-flight: Checking for exclusive node access"
    local other_gpu_procs
    other_gpu_procs=$(nvidia-smi --query-compute-apps=pid,process_name \
        --format=csv,noheader 2>/dev/null | grep -v "^$" || true)

    if [[ -n "${other_gpu_procs}" ]]; then
        log_error "GPU processes detected -- node is not exclusively allocated:"
        echo "${other_gpu_procs}" >&2
        check_fail "${CHECK_NAME}" \
            "Node is not exclusively allocated -- GPU processes running" "RESET"
        return 1
    fi

    # ── Pre-flight 2: Check MIG is disabled on all GPUs ──────────────────
    log_info "Pre-flight: Checking MIG status on all GPUs"
    local mig_output
    mig_output=$(nvidia-smi --query-gpu=index,mig.mode.current \
        --format=csv,noheader 2>/dev/null || echo "")

    if [[ -z "${mig_output}" ]]; then
        check_fail "${CHECK_NAME}" \
            "Unable to query MIG status" "RESET"
        return 1
    fi

    local mig_enabled_gpus=""
    while IFS=', ' read -r gpu_idx mig_status; do
        if [[ "${mig_status}" != "Disabled" && "${mig_status}" != "[N/A]" ]]; then
            mig_enabled_gpus+=" GPU${gpu_idx}(${mig_status})"
        fi
    done <<< "${mig_output}"

    if [[ -n "${mig_enabled_gpus}" ]]; then
        check_fail "${CHECK_NAME}" \
            "MIG is enabled on:${mig_enabled_gpus} -- EUD requires MIG disabled" "RESET"
        return 1
    fi
    log_verbose "MIG status: disabled on all GPUs"

    # ── Pre-flight 3: Stop concurrent GPU telemetry ──────────────────────
    log_info "Pre-flight: Stopping concurrent GPU telemetry services"
    if systemctl_available; then
        for svc in "${GPU_TELEMETRY_SERVICES[@]}"; do
            if systemctl is-active "${svc}.service" > /dev/null 2>&1; then
                log_warn "Stopping ${svc}.service for duration of L4 test"
                systemctl stop "${svc}.service" 2>/dev/null || true
                STOPPED_SERVICES+=("${svc}")
            fi
        done
    fi

    # Also check for dcgm-exporter running as container/process
    local exporter_pids
    exporter_pids=$(pgrep -f "dcgm-exporter" 2>/dev/null || true)
    if [[ -n "${exporter_pids}" ]]; then
        log_warn "Killing dcgm-exporter process(es): ${exporter_pids}"
        kill ${exporter_pids} 2>/dev/null || true
        sleep 2
    fi

    # ── Pre-flight 4: Verify nv-hostengine ───────────────────────────────
    log_info "Pre-flight: Checking nv-hostengine"
    if ! pgrep -x nv-hostengine > /dev/null 2>&1; then
        log_warn "nv-hostengine not running -- starting"
        if systemctl_available && systemctl is-enabled nvidia-dcgm.service > /dev/null 2>&1; then
            log_warn "DCGM managed by systemd -- be aware systemd may auto-restart services"
            log_warn "Consider: systemctl stop nvidia-dcgm && nv-hostengine"
        fi
        nv-hostengine 2>/dev/null || true
        sleep 2
    fi

    if ! pgrep -x nv-hostengine > /dev/null 2>&1; then
        check_fail "${CHECK_NAME}" "Unable to start nv-hostengine" "RESET"
        return 1
    fi

    # ── Execute DCGM Level 4 ─────────────────────────────────────────────
    log_info "Running DCGM Level 4 diagnostics (this will take 45+ minutes)"
    local dcgm_output
    local dcgm_exit=0

    dcgm_output=$(run_with_timeout "${DCGM_L4_TIMEOUT}" dcgmi diag -r 4 -j 2>&1) || dcgm_exit=$?

    # Save raw output regardless of exit code
    echo "${dcgm_output}" > "${RESULTS_DIR}/dcgm-l4-raw.json"

    if [[ ${dcgm_exit} -eq 124 ]]; then
        check_fail "${CHECK_NAME}" \
            "DCGM L4 diagnostics timed out after ${DCGM_L4_TIMEOUT}s" "RESET"
        return 1
    fi

    # ── Parse results ────────────────────────────────────────────────────
    local parse_result
    local parse_exit=0
    parse_result=$(echo "${dcgm_output}" | python3 "${SCRIPT_DIR}/../lib/parse-dcgm-results.py" \
        --level 4 2>&1) || parse_exit=$?

    echo "${parse_result}" > "${RESULTS_DIR}/check-${CHECK_NAME}.json"

    if [[ ${parse_exit} -ne 0 ]]; then
        check_fail "${CHECK_NAME}" "Failed to parse DCGM L4 results" "RESET"
        return 1
    fi

    # Extract severity
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
            check_pass "${CHECK_NAME}" "All DCGM L4 tests passed"
            return 0
            ;;
        FAIL)
            check_fail "${CHECK_NAME}" \
                "DCGM L4 diagnostics failed" "${overall_severity}"
            return 1
            ;;
        WARN)
            check_warn "${CHECK_NAME}" \
                "DCGM L4 completed with warnings (severity: ${overall_severity})"
            return 0
            ;;
        *)
            check_warn "${CHECK_NAME}" \
                "DCGM L4 returned unexpected status: ${overall_status}"
            return 0
            ;;
    esac
}

# ─── Entry point ─────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

run_check
