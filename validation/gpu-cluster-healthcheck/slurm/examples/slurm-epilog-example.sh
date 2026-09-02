#!/usr/bin/env bash
# Slurm Epilog Example -- Post-job GPU health check with exit-code routing
#
# Runs a quick nvidia-smi validation after each job completes.
# If the job failed with certain exit codes, triggers a lightweight health check.
#
# Configuration in slurm.conf:
#   Epilog=/path/to/slurm-epilog-example.sh
#   EpilogTimeout=300   # 5 minutes for quick check
#
# This script demonstrates exit-code routing:
#   - Jobs ending normally: quick nvidia-smi check only
#   - Jobs ending with GPU errors: full lightweight suite
#   - Jobs ending with NCCL errors: lightweight + EFA focus

set -euo pipefail

HEALTHCHECK_DIR="${HEALTHCHECK_DIR:-/shared/gpu-health-checks}"

log_epilog() {
    logger -t "gpu-healthcheck-epilog" "$*"
}

HOSTNAME=$(hostname)
JOB_EXIT_CODE="${SLURM_JOB_EXIT_CODE:-0}"
JOB_ID="${SLURM_JOB_ID:-unknown}"

log_epilog "Epilog starting for job ${JOB_ID} on ${HOSTNAME} (exit code: ${JOB_EXIT_CODE})"

# ─── Quick check: always run nvidia-smi ──────────────────────────────────────
NVIDIA_EXIT=0
nvidia-smi --query-gpu=gpu_name --format=csv,noheader > /dev/null 2>&1 || NVIDIA_EXIT=$?

if [[ ${NVIDIA_EXIT} -ne 0 ]]; then
    log_epilog "CRITICAL: nvidia-smi failed in epilog -- draining node ${HOSTNAME}"
    scontrol update NodeName="${HOSTNAME}" State=DRAIN Reason="Epilog: nvidia-smi failed"
    exit 1
fi

# ─── Exit-code routing ──────────────────────────────────────────────────────
# SLURM_JOB_EXIT_CODE2 provides "exit_code:signal" format directly,
# avoiding fragile bit-mask decoding of SLURM_JOB_EXIT_CODE.
IFS=':' read -r EXIT_CODE EXIT_SIGNAL <<< "${SLURM_JOB_EXIT_CODE2:-0:0}"

case ${EXIT_CODE} in
    0)
        # Normal exit -- no additional checks needed
        log_epilog "Job ${JOB_ID} exited normally -- nvidia-smi OK"
        ;;

    1|2|3)
        # Generic errors -- run quick Xid check
        log_epilog "Job ${JOB_ID} failed (exit ${EXIT_CODE}) -- checking for Xid errors"
        XID_ERRORS=$(dmesg --time-format iso 2>/dev/null | grep -i "NVRM.*Xid" | tail -5 || true)
        if [[ -n "${XID_ERRORS}" ]]; then
            log_epilog "Xid errors found after job failure -- triggering lightweight check"
            RESULTS_DIR="/tmp/gpu-healthcheck-epilog-${JOB_ID}"
            bash "${HEALTHCHECK_DIR}/checks/0-nvidia-smi-check.sh" || {
                log_epilog "nvidia-smi check failed -- draining node"
                scontrol update NodeName="${HOSTNAME}" State=DRAIN \
                    Reason="Epilog: Xid errors + nvidia-smi failure after job ${JOB_ID}"
                exit 1
            }
        fi
        ;;

    137|139|134)
        # Signal kills (SIGKILL=137, SIGSEGV=139, SIGABRT=134)
        # Potential GPU hang or driver issue
        log_epilog "Job ${JOB_ID} killed by signal (exit ${EXIT_CODE}) -- running lightweight check"
        RESULTS_DIR="/tmp/gpu-healthcheck-epilog-${JOB_ID}"
        export RESULTS_DIR
        bash "${HEALTHCHECK_DIR}/gpu-healthcheck.sh" --prolog || {
            log_epilog "Lightweight check failed -- draining node"
            scontrol update NodeName="${HOSTNAME}" State=DRAIN \
                Reason="Epilog: Health check failed after signal kill in job ${JOB_ID}"
            exit 1
        }
        ;;

    *)
        log_epilog "Job ${JOB_ID} exited with code ${EXIT_CODE} -- no special handling"
        ;;
esac

exit 0
