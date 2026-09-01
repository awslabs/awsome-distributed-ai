#!/usr/bin/env bash
# Slurm Prolog: GPU Health Check
# Runs fast checks (0, 2) before job execution by default.
# Non-zero exit causes Slurm to drain the node and requeue the job.
#
# Default behavior (~8 seconds):
#   Runs check 0 (nvidia-smi) and check 2 (EFA enumeration) only.
#   Prolog output goes to syslog / slurmd logs, not job output files.
#
# Configuration in slurm.conf:
#   Prolog=/path/to/prolog-gpu-healthcheck.sh
#   PrologTimeout=900   # 15 minutes
#
# Environment variable overrides:
#   GPU_HEALTHCHECK_PROLOG_ENABLE_DCGM=1  -- Enable DCGM L2 (check 1) in prolog

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEALTHCHECK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
HEALTHCHECK_SCRIPT="${HEALTHCHECK_DIR}/gpu-healthcheck.sh"

# Log to syslog for Slurm prolog visibility
log_prolog() {
    logger -t "gpu-healthcheck-prolog" "$*"
    echo "[gpu-healthcheck-prolog] $*" >&2
}

# ─── Main ────────────────────────────────────────────────────────────────────
main() {
    log_prolog "Starting GPU health check prolog for job ${SLURM_JOB_ID:-unknown}"
    log_prolog "Node: $(hostname), User: ${SLURM_JOB_USER:-unknown}"

    # Set results directory under /tmp with job context
    export RESULTS_DIR="/tmp/gpu-healthcheck-prolog-${SLURM_JOB_ID:-$$}"

    # DCGM L2 is off by default (adds minutes of silence before job output).
    # Set GPU_HEALTHCHECK_PROLOG_ENABLE_DCGM=1 to opt in.
    if [[ "${GPU_HEALTHCHECK_PROLOG_ENABLE_DCGM:-0}" != "1" ]]; then
        log_prolog "Running fast prolog (checks 0, 2 only; set GPU_HEALTHCHECK_PROLOG_ENABLE_DCGM=1 to include DCGM L2)"
        # Run only checks 0 and 2
        local exit_code=0
        bash "${HEALTHCHECK_DIR}/checks/0-nvidia-smi-check.sh" || exit_code=$?
        if [[ ${exit_code} -ne 0 ]]; then
            log_prolog "FAIL: nvidia-smi check failed -- draining node"
            exit 1
        fi

        bash "${HEALTHCHECK_DIR}/checks/2-efa-enumeration.sh" || exit_code=$?
        if [[ ${exit_code} -ne 0 ]]; then
            log_prolog "FAIL: EFA enumeration failed -- draining node"
            exit 1
        fi

        log_prolog "PASS: Prolog checks completed (DCGM skipped)"
        exit 0
    fi

    # Run full prolog suite (checks 0-2) including DCGM L2
    log_prolog "Running full prolog suite including DCGM L2 (GPU_HEALTHCHECK_PROLOG_ENABLE_DCGM=1)"
    local exit_code=0
    bash "${HEALTHCHECK_SCRIPT}" --prolog --results-dir "${RESULTS_DIR}" || exit_code=$?

    if [[ ${exit_code} -ne 0 ]]; then
        log_prolog "FAIL: Prolog health checks failed (exit ${exit_code}) -- draining node"
        log_prolog "Results: ${RESULTS_DIR}"
        exit 1
    fi

    log_prolog "PASS: All prolog health checks passed"

    # Clean up results on success (optional -- comment out to retain)
    rm -rf "${RESULTS_DIR}" 2>/dev/null || true

    exit 0
}

main "$@"
