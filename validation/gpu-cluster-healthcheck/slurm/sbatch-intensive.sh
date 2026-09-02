#!/usr/bin/env bash
#SBATCH --job-name=gpu-healthcheck-intensive
#SBATCH --output=gpu-healthcheck-intensive-%j.out
#SBATCH --error=gpu-healthcheck-intensive-%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#
# GPU Health Check -- Intensive Suite (sbatch)
# Runs checks 4-6 (DCGM L4, NCCL all_reduce, EFA loopback).
# REQUIRES exclusive node allocation -- no concurrent GPU workloads.
#
# Usage:
#   sbatch -N 2 sbatch-intensive.sh
#   sbatch -N 2 -p maintenance sbatch-intensive.sh
#
# ═══════════════════════════════════════════════════════════════════════════════
# User Variables -- Modify these as needed
# ═══════════════════════════════════════════════════════════════════════════════

# Path to the gpu-healthcheck directory
HEALTHCHECK_DIR="${HEALTHCHECK_DIR:-/shared/gpu-health-checks}"

# Results base directory
RESULTS_BASE="${RESULTS_BASE:-/shared/healthcheck-results}"

# NCCL test container image. The `docker://` scheme is required on older Pyxis
# (< v0.20); newer Pyxis treats it as a no-op. See checks/5-nccl-allreduce.sh
# for the rationale.
NCCL_CONTAINER="${NCCL_CONTAINER:-docker://public.ecr.aws/hpc-cloud/nccl-tests:latest}"

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

JOB_RESULTS_DIR="${RESULTS_BASE}/job-${SLURM_JOB_ID}-intensive"
mkdir -p "${JOB_RESULTS_DIR}"

echo "═══════════════════════════════════════════════════════════════"
echo "GPU Health Check -- Intensive Suite"
echo "═══════════════════════════════════════════════════════════════"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Nodes:     ${SLURM_JOB_NUM_NODES}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Results:   ${JOB_RESULTS_DIR}"
echo ""
echo "WARNING: This suite includes DCGM Level 4 diagnostics."
echo "  - Runtime: 45 min - 2.25 hr per node"
echo "  - Pulse power test causes variable power draw"
echo "  - MIG must be disabled"
echo "  - No concurrent GPU telemetry during test"
echo "═══════════════════════════════════════════════════════════════"
echo ""

export HEALTHCHECK_DIR NCCL_CONTAINER

# Run per-node intensive checks (4 and 6) via srun
# Check 5 (NCCL) runs separately as it requires multi-node coordination
# Disable set -e around srun so that node failures don't prevent aggregation.
set +e
srun --ntasks-per-node=1 bash -c '
    set -euo pipefail
    HOSTNAME=$(hostname)
    NODE_RESULTS_DIR="'"${JOB_RESULTS_DIR}"'/${HOSTNAME}"
    mkdir -p "${NODE_RESULTS_DIR}"
    export RESULTS_DIR="${NODE_RESULTS_DIR}"

    echo "[${HOSTNAME}] Starting intensive per-node checks (DCGM L4, EFA loopback)"

    # Run DCGM L4 (check 4)
    EXIT_CODE=0
    bash "'"${HEALTHCHECK_DIR}"'/checks/4-dcgm-diag-l4.sh" || EXIT_CODE=$?
    echo "[${HOSTNAME}] DCGM L4 completed (exit: ${EXIT_CODE})"

    # Run EFA loopback (check 6)
    bash "'"${HEALTHCHECK_DIR}"'/checks/6-efa-loopback.sh" || EXIT_CODE=$?
    echo "[${HOSTNAME}] EFA loopback completed (exit: ${EXIT_CODE})"

    exit ${EXIT_CODE}
'
PER_NODE_EXIT=$?
set -e

# Run multi-node NCCL test (check 5) if more than 1 node
NCCL_EXIT=0
if [[ "${SLURM_JOB_NUM_NODES}" -ge 2 ]]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "Running multi-node NCCL all_reduce test"
    echo "═══════════════════════════════════════════════════════════════"

    export RESULTS_DIR="${JOB_RESULTS_DIR}/nccl"
    mkdir -p "${RESULTS_DIR}"

    bash "${HEALTHCHECK_DIR}/checks/5-nccl-allreduce.sh" || NCCL_EXIT=$?
else
    echo ""
    echo "[INFO] Skipping NCCL all_reduce (requires >= 2 nodes, have ${SLURM_JOB_NUM_NODES})"
fi

# Aggregate results
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Aggregating results"
echo "═══════════════════════════════════════════════════════════════"

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

# Use boolean exit: 1 if either phase failed, 0 if both passed.
# Avoid arithmetic addition which can produce exit codes >125 (reserved by bash).
if [[ ${PER_NODE_EXIT} -ne 0 || ${NCCL_EXIT} -ne 0 ]]; then
    FINAL_EXIT=1
else
    FINAL_EXIT=0
fi
echo ""
echo "Full results: ${JOB_RESULTS_DIR}"
exit ${FINAL_EXIT}
