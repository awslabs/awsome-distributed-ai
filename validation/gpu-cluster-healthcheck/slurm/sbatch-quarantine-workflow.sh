#!/usr/bin/env bash
#SBATCH --job-name=gpu-quarantine-workflow
#SBATCH --output=gpu-quarantine-%j.out
#SBATCH --error=gpu-quarantine-%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=05:00:00
#
# GPU Quarantine Decision Workflow
# Implements the full health check decision flow:
#   drain → lightweight → severity → intensive → replace/return
#
# Designed for operators to run against suspected-faulty nodes.
#
# Usage:
#   sbatch -N 1 -w <suspect-node> sbatch-quarantine-workflow.sh
#
# ═══════════════════════════════════════════════════════════════════════════════
# User Variables
# ═══════════════════════════════════════════════════════════════════════════════

HEALTHCHECK_DIR="${HEALTHCHECK_DIR:-/shared/gpu-health-checks}"
RESULTS_BASE="${RESULTS_BASE:-/shared/healthcheck-results}"

# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

HEALTHCHECK_SCRIPT="${HEALTHCHECK_DIR}/gpu-healthcheck.sh"
PARSE_SCRIPT="${HEALTHCHECK_DIR}/lib/parse-dcgm-results.py"

if [[ ! -f "${HEALTHCHECK_SCRIPT}" ]]; then
    echo "ERROR: Health check script not found: ${HEALTHCHECK_SCRIPT}"
    exit 1
fi

JOB_RESULTS_DIR="${RESULTS_BASE}/quarantine-${SLURM_JOB_ID}"
mkdir -p "${JOB_RESULTS_DIR}"

HOSTNAME=$(hostname)
NODE_RESULTS="${JOB_RESULTS_DIR}/${HOSTNAME}"

echo "═══════════════════════════════════════════════════════════════"
echo "GPU Quarantine Decision Workflow"
echo "═══════════════════════════════════════════════════════════════"
echo "Node:      ${HOSTNAME}"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Results:   ${JOB_RESULTS_DIR}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ─── Phase 1: Lightweight Suite ──────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Phase 1: Lightweight Suite (checks 0-3)                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

LIGHTWEIGHT_DIR="${NODE_RESULTS}/lightweight"
mkdir -p "${LIGHTWEIGHT_DIR}"

LIGHTWEIGHT_EXIT=0
bash "${HEALTHCHECK_SCRIPT}" \
    --suite lightweight \
    --results-dir "${LIGHTWEIGHT_DIR}" \
    --json \
    || LIGHTWEIGHT_EXIT=$?

echo ""

if [[ ${LIGHTWEIGHT_EXIT} -eq 0 ]]; then
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  RESULT: Lightweight suite PASSED                          ║"
    echo "║  RECOMMENDATION: Return node to service                    ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    cat > "${JOB_RESULTS_DIR}/recommendation.json" <<ENDJSON
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "hostname": "${HOSTNAME}",
  "phase_completed": "lightweight",
  "overall_result": "PASS",
  "recommendation": "RETURN_TO_SERVICE",
  "action": "Resume node in Slurm: scontrol update NodeName=${HOSTNAME} State=RESUME",
  "evidence": "${LIGHTWEIGHT_DIR}"
}
ENDJSON

    echo ""
    echo "Action: scontrol update NodeName=${HOSTNAME} State=RESUME"
    echo "Results: ${JOB_RESULTS_DIR}"
    exit 0
fi

# ─── Analyze lightweight failure severity ────────────────────────────────────
echo "Lightweight suite FAILED -- analyzing severity"
echo ""

# Read severity from the summary
SEVERITY="ISOLATE"
if [[ -f "${LIGHTWEIGHT_DIR}/summary.json" ]]; then
    SEVERITY=$(python3 -c "
import json, sys
try:
    with open('${LIGHTWEIGHT_DIR}/summary.json') as f:
        data = json.load(f)
    # Check individual check results for severity
    import glob, os
    max_sev = 0
    sev_map = {'ISOLATE': 4, 'REBOOT': 3, 'RESET': 2, 'MONITOR': 1, 'PASS': 0}
    rev_map = {4: 'ISOLATE', 3: 'REBOOT', 2: 'RESET', 1: 'MONITOR', 0: 'PASS'}
    for f in glob.glob(os.path.join('${LIGHTWEIGHT_DIR}', 'check-*.json')):
        with open(f) as fh:
            r = json.load(fh)
            s = r.get('severity', r.get('overall_severity', ''))
            max_sev = max(max_sev, sev_map.get(s, 0))
    print(rev_map.get(max_sev, 'ISOLATE'))
except Exception:
    print('ISOLATE')
" 2>/dev/null || echo "ISOLATE")
fi

echo "Detected severity: ${SEVERITY}"

case "${SEVERITY}" in
    ISOLATE)
        echo ""
        echo "╔══════════════════════════════════════════════════════════════╗"
        echo "║  SEVERITY: ISOLATE -- Critical hardware fault               ║"
        echo "║  RECOMMENDATION: Replace instance                          ║"
        echo "╚══════════════════════════════════════════════════════════════╝"

        cat > "${JOB_RESULTS_DIR}/recommendation.json" <<ENDJSON
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "hostname": "${HOSTNAME}",
  "phase_completed": "lightweight",
  "overall_result": "FAIL",
  "severity": "ISOLATE",
  "recommendation": "REPLACE_INSTANCE",
  "action": "Replace instance -- do not return to service",
  "evidence": "${LIGHTWEIGHT_DIR}"
}
ENDJSON

        echo ""
        echo "Action: Replace instance (do not attempt repair)"
        echo "Results: ${JOB_RESULTS_DIR}"
        exit 1
        ;;

    REBOOT)
        echo ""
        echo "╔══════════════════════════════════════════════════════════════╗"
        echo "║  SEVERITY: REBOOT -- Node reboot required                   ║"
        echo "║  Proceeding to intensive suite after reboot check           ║"
        echo "╚══════════════════════════════════════════════════════════════╝"

        cat > "${JOB_RESULTS_DIR}/recommendation.json" <<ENDJSON
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "hostname": "${HOSTNAME}",
  "phase_completed": "lightweight",
  "overall_result": "FAIL",
  "severity": "REBOOT",
  "recommendation": "REBOOT_NODE",
  "action": "scontrol reboot nextstate=resume ${HOSTNAME}",
  "evidence": "${LIGHTWEIGHT_DIR}"
}
ENDJSON

        echo ""
        echo "Severity: REBOOT -- proceeding to intensive suite after reboot check"
        echo ""
        ;;

    MONITOR)
        echo ""
        echo "╔══════════════════════════════════════════════════════════════╗"
        echo "║  SEVERITY: MONITOR -- Minor issue detected                  ║"
        echo "║  RECOMMENDATION: Return to service, flag for review        ║"
        echo "╚══════════════════════════════════════════════════════════════╝"

        cat > "${JOB_RESULTS_DIR}/recommendation.json" <<ENDJSON
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "hostname": "${HOSTNAME}",
  "phase_completed": "lightweight",
  "overall_result": "WARN",
  "severity": "MONITOR",
  "recommendation": "RETURN_WITH_FLAG",
  "action": "Resume node with monitoring flag: scontrol update NodeName=${HOSTNAME} State=RESUME Comment='Health-check-monitor'",
  "evidence": "${LIGHTWEIGHT_DIR}"
}
ENDJSON

        echo ""
        echo "Action: scontrol update NodeName=${HOSTNAME} State=RESUME Comment='Health-check-monitor'"
        echo "Results: ${JOB_RESULTS_DIR}"
        exit 0
        ;;

    RESET)
        echo ""
        echo "Severity: RESET -- proceeding to intensive suite after reboot check"
        echo ""
        ;;
esac

# ─── Phase 2: Intensive Suite ────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Phase 2: Intensive Suite (checks 4-6)                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

INTENSIVE_DIR="${NODE_RESULTS}/intensive"
mkdir -p "${INTENSIVE_DIR}"

INTENSIVE_EXIT=0
bash "${HEALTHCHECK_SCRIPT}" \
    --suite intensive \
    --exclusive \
    --results-dir "${INTENSIVE_DIR}" \
    --json \
    || INTENSIVE_EXIT=$?

echo ""

if [[ ${INTENSIVE_EXIT} -eq 0 ]]; then
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  RESULT: Intensive suite PASSED                            ║"
    echo "║  RECOMMENDATION: Return node to service                    ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    cat > "${JOB_RESULTS_DIR}/recommendation.json" <<ENDJSON
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "hostname": "${HOSTNAME}",
  "phase_completed": "intensive",
  "overall_result": "PASS",
  "recommendation": "RETURN_TO_SERVICE",
  "action": "Resume node in Slurm: scontrol update NodeName=${HOSTNAME} State=RESUME",
  "evidence_lightweight": "${LIGHTWEIGHT_DIR}",
  "evidence_intensive": "${INTENSIVE_DIR}"
}
ENDJSON

    echo ""
    echo "Action: scontrol update NodeName=${HOSTNAME} State=RESUME"
else
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  RESULT: Intensive suite FAILED                            ║"
    echo "║  RECOMMENDATION: Replace instance                          ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    cat > "${JOB_RESULTS_DIR}/recommendation.json" <<ENDJSON
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "hostname": "${HOSTNAME}",
  "phase_completed": "intensive",
  "overall_result": "FAIL",
  "recommendation": "REPLACE_INSTANCE",
  "action": "Replace instance -- confirmed hardware fault",
  "evidence_lightweight": "${LIGHTWEIGHT_DIR}",
  "evidence_intensive": "${INTENSIVE_DIR}"
}
ENDJSON

    echo ""
    echo "Action: Replace instance (confirmed hardware fault)"
fi

echo "Results: ${JOB_RESULTS_DIR}"
echo ""
cat "${JOB_RESULTS_DIR}/recommendation.json"
exit ${INTENSIVE_EXIT}
