#!/usr/bin/env bash
# agent.sh -- DaemonSet entrypoint for GPU health check agent
#
# Runs lightweight checks in a loop, patches Node labels/taints based on
# results, and emits JSON summaries to stdout for log collection.
#
# Required env:
#   NODE_NAME          -- Kubernetes node name (from downward API)
#
# Optional env:
#   CHECK_INTERVAL     -- Seconds between lightweight check cycles (default: 300)
#   DCGM_L2_INTERVAL   -- Seconds between DCGM L2 runs (default: 21600 / 6hr)
#   CHECKS_LIGHTWEIGHT -- Space-separated check numbers (default: "0 2 3")
#   ENABLE_TAINT       -- Set to "false" to disable taint management (default: true)
#   ENABLE_LABEL       -- Set to "false" to disable label management (default: true)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=../lib/common.sh
source "${BASE_DIR}/lib/common.sh"

# ─── Configuration ──────────────────────────────────────────────────────────
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"
DCGM_L2_INTERVAL="${DCGM_L2_INTERVAL:-21600}"
CHECKS_LIGHTWEIGHT="${CHECKS_LIGHTWEIGHT:-0 2 3}"
ENABLE_TAINT="${ENABLE_TAINT:-true}"
ENABLE_LABEL="${ENABLE_LABEL:-true}"

LABEL_PREFIX="gpu-healthcheck.aws-samples.io"
TAINT_KEY="${LABEL_PREFIX}/unhealthy"

RESULTS_DIR="${RESULTS_DIR:-/tmp/gpu-healthcheck-agent}"

# Track last DCGM L2 run time (epoch seconds)
LAST_L2_RUN=0

# ─── Helpers ────────────────────────────────────────────────────────────────

die() {
    log_error "$*"
    exit 1
}

require_env() {
    local var="$1"
    if [[ -z "${!var:-}" ]]; then
        die "Required environment variable ${var} is not set"
    fi
}

# Detect instance type: IMDS first, then K8s node label fallback
detect_instance() {
    # Try IMDS (works with hostNetwork: true)
    INSTANCE_TYPE=$(detect_instance_type 2>/dev/null || true)

    if [[ -z "${INSTANCE_TYPE}" ]]; then
        log_info "IMDS unavailable, falling back to node label"
        INSTANCE_TYPE=$(kubectl get node "${NODE_NAME}" \
            -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type}' 2>/dev/null || true)
    fi

    if [[ -z "${INSTANCE_TYPE}" ]]; then
        log_warn "Unable to detect instance type -- using defaults"
        INSTANCE_TYPE="unknown"
    fi

    log_info "Instance type: ${INSTANCE_TYPE}"
}

# Patch Node labels
patch_labels() {
    local status="$1"
    local severity="$2"
    local timestamp
    # K8s label values cannot contain colons -- use compact ISO 8601 basic format
    timestamp="$(date -u +"%Y-%m-%dT%H%M%SZ")"

    if [[ "${ENABLE_LABEL}" != "true" ]]; then
        return 0
    fi

    kubectl label node "${NODE_NAME}" \
        "${LABEL_PREFIX}/status=${status}" \
        "${LABEL_PREFIX}/severity=${severity}" \
        "${LABEL_PREFIX}/last-check=${timestamp}" \
        --overwrite 2>/dev/null || log_warn "Failed to patch node labels"
}

# Add unhealthy taint
add_taint() {
    if [[ "${ENABLE_TAINT}" != "true" ]]; then
        return 0
    fi

    if ! kubectl get node "${NODE_NAME}" -o jsonpath='{.spec.taints}' 2>/dev/null \
            | grep -q "${TAINT_KEY}"; then
        kubectl taint node "${NODE_NAME}" \
            "${TAINT_KEY}=true:NoSchedule" 2>/dev/null \
            || log_warn "Failed to add unhealthy taint"
        log_warn "Added NoSchedule taint: ${TAINT_KEY}"
    fi
}

# Remove unhealthy taint
remove_taint() {
    if [[ "${ENABLE_TAINT}" != "true" ]]; then
        return 0
    fi

    kubectl taint node "${NODE_NAME}" \
        "${TAINT_KEY}=true:NoSchedule-" 2>/dev/null || true
}

# Annotate Node with compact JSON summary
annotate_node() {
    local summary="$1"

    kubectl annotate node "${NODE_NAME}" \
        "${LABEL_PREFIX}/results=${summary}" \
        --overwrite 2>/dev/null || log_warn "Failed to annotate node"
}

# Determine severity from check result files
determine_severity() {
    python3 "${SCRIPT_DIR}/determine-severity.py" \
        --results-dir "${RESULTS_DIR}" 2>/dev/null || echo "fail:RESET"
}

# ─── Check Runner ───────────────────────────────────────────────────────────

run_checks() {
    local now
    now=$(date +%s)
    local run_l2=false

    # Should we run DCGM L2 this cycle?
    local elapsed=$(( now - LAST_L2_RUN ))
    if [[ ${elapsed} -ge ${DCGM_L2_INTERVAL} ]]; then
        run_l2=true
    fi

    # Clean previous results
    rm -f "${RESULTS_DIR}"/check-*.json

    # Run lightweight checks
    local check_exit=0
    for check_num in ${CHECKS_LIGHTWEIGHT}; do
        local script="${BASE_DIR}/checks/${check_num}-"*.sh
        # Expand glob -- there should be exactly one match
        local scripts=( ${script} )
        if [[ -f "${scripts[0]}" ]]; then
            log_info "Running check ${check_num}"
            bash "${scripts[0]}" || check_exit=$?
        else
            log_warn "Check script not found for check ${check_num}"
        fi
    done

    # Run DCGM L2 if interval has elapsed
    if [[ "${run_l2}" == "true" ]]; then
        log_info "Running DCGM L2 (interval: ${DCGM_L2_INTERVAL}s)"
        bash "${BASE_DIR}/checks/1-dcgm-diag-l2.sh" || check_exit=$?
        LAST_L2_RUN=$(date +%s)
    fi

    # Aggregate severity
    local result
    result=$(determine_severity)
    local status="${result%%:*}"
    local severity="${result##*:}"

    # Patch Node state
    patch_labels "${status}" "${severity}"

    if [[ "${status}" == "pass" ]]; then
        remove_taint
    else
        add_taint
    fi

    # Build compact summary for annotation
    local summary
    summary=$(python3 -c "
import json, os, glob, socket
from datetime import datetime, timezone
checks = []
for f in sorted(glob.glob('${RESULTS_DIR}/check-*.json')):
    try:
        with open(f) as fh:
            checks.append(json.load(fh))
    except Exception:
        pass
print(json.dumps({
    'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    'hostname': socket.gethostname(),
    'instance_type': '${INSTANCE_TYPE}',
    'status': '${status}',
    'severity': '${severity}',
    'checks': [{
        'check': c.get('check', ''),
        'status': c.get('status', ''),
        'severity': c.get('severity', ''),
    } for c in checks],
}, separators=(',', ':')))
" 2>/dev/null || echo '{}')

    annotate_node "${summary}"

    # Emit JSON summary to stdout for log collection
    echo "${summary}"

    return ${check_exit}
}

# ─── Signal Handlers ────────────────────────────────────────────────────────

cleanup() {
    log_info "Received shutdown signal -- cleaning up"
    remove_taint
    if [[ "${ENABLE_LABEL}" == "true" ]]; then
        kubectl label node "${NODE_NAME}" \
            "${LABEL_PREFIX}/status=agent-stopped" \
            --overwrite 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGTERM SIGINT

# ─── Main Loop ──────────────────────────────────────────────────────────────

main() {
    require_env NODE_NAME

    log_info "GPU Health Check Agent starting"
    log_info "Node: ${NODE_NAME}"
    log_info "Check interval: ${CHECK_INTERVAL}s, DCGM L2 interval: ${DCGM_L2_INTERVAL}s"
    log_info "Lightweight checks: ${CHECKS_LIGHTWEIGHT}"
    log_info "Taint management: ${ENABLE_TAINT}, Label management: ${ENABLE_LABEL}"

    ensure_results_dir
    detect_instance
    load_instance_profile || true

    while true; do
        log_info "Starting health check cycle"
        run_checks || true
        log_info "Cycle complete, sleeping ${CHECK_INTERVAL}s"

        # Sleep in a way that is interruptible by SIGTERM
        sleep "${CHECK_INTERVAL}" &
        wait $! || true
    done
}

main "$@"
