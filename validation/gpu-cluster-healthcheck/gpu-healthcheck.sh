#!/usr/bin/env bash
# gpu-healthcheck.sh -- Master orchestrator for GPU cluster health checks
#
# Provides two operational modes:
#   lightweight (checks 0-3): Regular use -- prolog/epilog, cron sweeps
#   intensive   (checks 4-6): Quarantine/post-mortem -- exclusive node access
#
# Usage:
#   gpu-healthcheck.sh --suite lightweight
#   gpu-healthcheck.sh --suite intensive --exclusive
#   gpu-healthcheck.sh --check 1
#   gpu-healthcheck.sh --prolog
#   gpu-healthcheck.sh --suite lightweight --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

# ─── Check Registry ─────────────────────────────────────────────────────────
# Indexed arrays: position corresponds to check number 0-6
CHECK_SCRIPTS=(
    "${SCRIPT_DIR}/checks/0-nvidia-smi-check.sh"
    "${SCRIPT_DIR}/checks/1-dcgm-diag-l2.sh"
    "${SCRIPT_DIR}/checks/2-efa-enumeration.sh"
    "${SCRIPT_DIR}/checks/3-topology-check.sh"
    "${SCRIPT_DIR}/checks/4-dcgm-diag-l4.sh"
    "${SCRIPT_DIR}/checks/5-nccl-allreduce.sh"
    "${SCRIPT_DIR}/checks/6-efa-loopback.sh"
)

CHECK_NAMES=(
    "nvidia-smi"
    "dcgm-l2"
    "efa-enumeration"
    "topology"
    "dcgm-l4"
    "nccl-allreduce"
    "efa-loopback"
)

LIGHTWEIGHT_CHECKS=(0 1 2 3)
INTENSIVE_CHECKS=(4 5 6)
PROLOG_CHECKS=(0 1 2)

# ─── Defaults ────────────────────────────────────────────────────────────────
SUITE=""
SINGLE_CHECK=""
PROLOG_MODE=0
EXCLUSIVE_CONFIRMED=0
CHECK_TIMEOUT="${CHECK_TIMEOUT:-900}"  # 15-minute default per-check timeout

# ─── Usage ───────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
GPU Cluster Health Check Suite

Usage: $(basename "$0") [OPTIONS]

Options:
  --suite lightweight|intensive    Run predefined check suite
  --check <N|name>                 Run individual check (0-6 or by name)
  --prolog                         Prolog mode: checks 0-2, fail-fast with non-zero exit
  --exclusive                      Confirm exclusive node access (required for intensive)
  --results-dir <path>             Output directory (default: /tmp/gpu-healthcheck-<ts>)
  --timeout <seconds>              Per-check timeout (default: ${CHECK_TIMEOUT})
  --json                           Output results as JSON
  --dry-run                        Print commands without executing
  -v, --verbose                    Verbose output
  -h, --help                       Show this help

Suites:
  lightweight   Checks 0-3: nvidia-smi, DCGM L2, EFA, topology (~15 min)
  intensive     Checks 4-6: DCGM L4, NCCL, EFA loopback (~1-3 hr, exclusive)

Examples:
  $(basename "$0") --suite lightweight
  $(basename "$0") --suite intensive --exclusive
  $(basename "$0") --check dcgm-l2
  $(basename "$0") --prolog
  $(basename "$0") --suite lightweight --dry-run
EOF
    exit 0
}

# ─── Argument Parsing ───────────────────────────────────────────────────────
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --suite)
                SUITE="$2"
                shift 2
                ;;
            --check)
                SINGLE_CHECK="$2"
                shift 2
                ;;
            --prolog)
                PROLOG_MODE=1
                shift
                ;;
            --exclusive)
                EXCLUSIVE_CONFIRMED=1
                shift
                ;;
            --results-dir)
                RESULTS_DIR="$2"
                shift 2
                ;;
            --timeout)
                CHECK_TIMEOUT="$2"
                shift 2
                ;;
            --json)
                JSON_OUTPUT=1
                shift
                ;;
            --dry-run)
                DRY_RUN=1
                shift
                ;;
            -v|--verbose)
                VERBOSE=1
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done
}

# ─── Resolve check number from name ─────────────────────────────────────────
resolve_check() {
    local input="$1"

    # If numeric, return directly
    if [[ "${input}" =~ ^[0-6]$ ]]; then
        echo "${input}"
        return 0
    fi

    # Search by name
    local num
    for num in 0 1 2 3 4 5 6; do
        if [[ "${CHECK_NAMES[$num]}" == "${input}" ]]; then
            echo "${num}"
            return 0
        fi
    done

    log_error "Unknown check: ${input}"
    log_error "Valid checks: 0-6 (or by name: ${CHECK_NAMES[*]})"
    return 1
}

# ─── Run a single check ─────────────────────────────────────────────────────
run_single_check() {
    local check_num="$1"
    local script="${CHECK_SCRIPTS[$check_num]}"
    local name="${CHECK_NAMES[$check_num]}"

    if [[ ! -f "${script}" ]]; then
        log_error "Check script not found: ${script}"
        return 1
    fi

    log_info "━━━ Check ${check_num}: ${name} ━━━"

    local check_exit=0
    local dry_run_flag=""
    [[ "${DRY_RUN}" == "1" ]] && dry_run_flag="--dry-run"

    # Export shared state for check scripts
    export RESULTS_DIR VERBOSE DRY_RUN JSON_OUTPUT
    export INSTANCE_TYPE EXPECTED_GPU_COUNT EXPECTED_EFA_COUNT NVLINK_EXPECTED EFA_PROVIDER

    if [[ "${DRY_RUN}" == "1" ]]; then
        bash "${script}" --dry-run || check_exit=$?
    else
        timeout "${CHECK_TIMEOUT}" bash "${script}" || check_exit=$?

        if [[ ${check_exit} -eq 124 ]]; then
            check_fail "${name}" "Check timed out after ${CHECK_TIMEOUT}s" "RESET"
            return 1
        fi
    fi

    return ${check_exit}
}

# ─── Run a suite of checks ──────────────────────────────────────────────────
run_suite() {
    local suite_var="$1"
    local suite_name="$2"
    local fail_fast="${3:-1}"

    # Copy array by name (portable alternative to nameref)
    eval 'local checks=("${'${suite_var}'[@]}")'

    log_info "Running ${suite_name} suite (${#checks[@]} checks)"
    echo ""

    local total_failures=0
    local total_checks=${#checks[@]}
    local completed=0

    for check_num in "${checks[@]}"; do
        local check_exit=0
        run_single_check "${check_num}" || check_exit=$?

        completed=$((completed + 1))

        if [[ ${check_exit} -ne 0 ]]; then
            total_failures=$((total_failures + 1))

            if [[ "${fail_fast}" == "1" && "${suite_name}" != "intensive" ]]; then
                log_error "Check ${check_num} failed -- stopping suite (fail-fast mode)"
                break
            fi
        fi

        echo ""
    done

    # Write summary
    write_summary "${suite_name}" "${total_checks}" "${completed}" "${total_failures}"

    return ${total_failures}
}

# ─── Write summary JSON ─────────────────────────────────────────────────────
write_summary() {
    local suite_name="$1"
    local total="$2"
    local completed="$3"
    local failures="$4"

    local status="PASS"
    [[ ${failures} -gt 0 ]] && status="FAIL"

    local summary
    summary=$(cat <<ENDJSON
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "hostname": "$(hostname)",
  "instance_type": "${INSTANCE_TYPE}",
  "suite": "${suite_name}",
  "total_checks": ${total},
  "completed_checks": ${completed},
  "failures": ${failures},
  "status": "${status}",
  "results_dir": "${RESULTS_DIR}"
}
ENDJSON
)

    echo "${summary}" > "${RESULTS_DIR}/summary.json"

    echo ""
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [[ "${status}" == "PASS" ]]; then
        echo -e "${GREEN}${BOLD}SUITE RESULT: PASS${NC} (${completed}/${total} checks passed)"
    else
        echo -e "${RED}${BOLD}SUITE RESULT: FAIL${NC} (${failures}/${total} checks failed)"
    fi
    log_info "Results: ${RESULTS_DIR}"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ─── Main ────────────────────────────────────────────────────────────────────
main() {
    parse_args "$@"

    # Initialize
    ensure_results_dir
    detect_instance_type > /dev/null 2>&1 || true
    load_instance_profile || true

    # Pre-flight: verify critical dependencies (python3) and log versions
    if ! preflight_checks; then
        log_error "Pre-flight checks failed -- cannot continue"
        exit 1
    fi

    log_info "GPU Health Check Suite"
    log_info "Host: $(hostname) | Instance: ${INSTANCE_TYPE:-unknown}"
    log_info "Results: ${RESULTS_DIR}"
    [[ "${DRY_RUN}" == "1" ]] && log_warn "DRY-RUN MODE -- no commands will be executed"
    echo ""

    # Route to appropriate mode
    if [[ "${PROLOG_MODE}" == "1" ]]; then
        run_suite PROLOG_CHECKS "prolog" 1
        exit $?
    fi

    if [[ -n "${SINGLE_CHECK}" ]]; then
        local check_num
        check_num=$(resolve_check "${SINGLE_CHECK}")
        run_single_check "${check_num}"
        exit $?
    fi

    case "${SUITE}" in
        lightweight)
            run_suite LIGHTWEIGHT_CHECKS "lightweight" 1
            exit $?
            ;;
        intensive)
            if [[ "${EXCLUSIVE_CONFIRMED}" != "1" && "${DRY_RUN}" != "1" ]]; then
                log_error "Intensive suite requires --exclusive flag to confirm exclusive node access"
                log_error "This suite includes DCGM L4 which requires no other GPU workloads"
                exit 1
            fi
            run_suite INTENSIVE_CHECKS "intensive" 0
            exit $?
            ;;
        "")
            log_error "No mode specified. Use --suite, --check, or --prolog"
            echo ""
            usage
            ;;
        *)
            log_error "Unknown suite: ${SUITE}"
            usage
            ;;
    esac
}

main "$@"
