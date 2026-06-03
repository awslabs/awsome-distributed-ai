#!/usr/bin/env bash
#SBATCH --job-name=gpu-healthcheck-lightweight
#SBATCH --output=gpu-healthcheck-lightweight-%j.out
#SBATCH --error=gpu-healthcheck-lightweight-%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#
# GPU Health Check -- Lightweight Suite (sbatch)
# Runs checks 0-3 (nvidia-smi, DCGM L2, EFA, topology) across all allocated nodes.
#
# Usage:
#   sbatch -N 4 sbatch-lightweight.sh
#   sbatch -N 4 -p gpu sbatch-lightweight.sh
#
# ═══════════════════════════════════════════════════════════════════════════════
# User Variables -- Modify these as needed
# ═══════════════════════════════════════════════════════════════════════════════

# Path to the gpu-healthcheck directory
HEALTHCHECK_DIR="${HEALTHCHECK_DIR:-/shared/gpu-health-checks}"

# Results base directory (per-node results stored under this)
RESULTS_BASE="${RESULTS_BASE:-/shared/healthcheck-results}"

# Skip DCGM L2 for faster execution (set to 1 to skip)
SKIP_DCGM="${SKIP_DCGM:-0}"

# ═══════════════════════════════════════════════════════════════════════════════
# End User Variables
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

HEALTHCHECK_SCRIPT="${HEALTHCHECK_DIR}/gpu-healthcheck.sh"

if [[ ! -f "${HEALTHCHECK_SCRIPT}" ]]; then
    echo "ERROR: Health check script not found: ${HEALTHCHECK_SCRIPT}"
    echo "Set HEALTHCHECK_DIR to the correct path"
    exit 1
fi

# Create per-job results directory
JOB_RESULTS_DIR="${RESULTS_BASE}/job-${SLURM_JOB_ID}"
mkdir -p "${JOB_RESULTS_DIR}"

echo "═══════════════════════════════════════════════════════════════"
echo "GPU Health Check -- Lightweight Suite"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Nodes:     ${SLURM_JOB_NUM_NODES}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Results:   ${JOB_RESULTS_DIR}"
echo "═══════════════════════════════════════════════════════════════"

# Run health checks on each node via srun
export HEALTHCHECK_DIR SKIP_DCGM

# Disable set -e around srun so that node failures don't prevent aggregation.
# The exit code is captured explicitly and propagated at the end.
set +e
srun --ntasks-per-node=1 bash -c '
    set -euo pipefail
    HOSTNAME=$(hostname)
    NODE_RESULTS_DIR="'"${JOB_RESULTS_DIR}"'/${HOSTNAME}"
    mkdir -p "${NODE_RESULTS_DIR}"

    echo "[${HOSTNAME}] Starting lightweight health checks"

    EXIT_CODE=0
    bash "'"${HEALTHCHECK_SCRIPT}"'" \
        --suite lightweight \
        --results-dir "${NODE_RESULTS_DIR}" \
        --json \
        || EXIT_CODE=$?

    echo "[${HOSTNAME}] Completed with exit code: ${EXIT_CODE}"
    exit ${EXIT_CODE}
'
SRUN_EXIT=$?
set -e

# Aggregate results
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Aggregating results across nodes"
echo "═══════════════════════════════════════════════════════════════"

# Collect per-node result directories
NODE_DIRS=()
for node_dir in "${JOB_RESULTS_DIR}"/*/; do
    [[ -d "${node_dir}" ]] && NODE_DIRS+=("${node_dir}")
done

if [[ ${#NODE_DIRS[@]} -gt 0 ]]; then
    python3 "${HEALTHCHECK_DIR}/lib/aggregate-results.py" \
        --results-dir "${NODE_DIRS[@]}" \
        --format table \
        --output "${JOB_RESULTS_DIR}/cluster-summary.txt"

    python3 "${HEALTHCHECK_DIR}/lib/aggregate-results.py" \
        --results-dir "${NODE_DIRS[@]}" \
        --format json \
        --output "${JOB_RESULTS_DIR}/cluster-summary.json"

    echo ""
    cat "${JOB_RESULTS_DIR}/cluster-summary.txt"
fi

echo ""
echo "Full results: ${JOB_RESULTS_DIR}"
exit ${SRUN_EXIT}
