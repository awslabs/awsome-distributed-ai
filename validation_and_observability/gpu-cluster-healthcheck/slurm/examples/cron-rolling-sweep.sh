#!/usr/bin/env bash
# Cron Rolling Sweep -- Periodic GPU health check across idle Slurm nodes
#
# Submits lightweight health checks to idle nodes in a rolling fashion,
# checking a configurable number of nodes per sweep.
#
# Crontab example (run every 4 hours):
#   0 */4 * * * /path/to/cron-rolling-sweep.sh >> /var/log/gpu-healthcheck-sweep.log 2>&1
#
# Configuration:
#   NODES_PER_SWEEP -- Number of idle nodes to check per sweep (default: 10)
#   HEALTHCHECK_DIR -- Path to the gpu-health-checks directory
#   RESULTS_BASE    -- Base directory for results

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
HEALTHCHECK_DIR="${HEALTHCHECK_DIR:-/shared/gpu-health-checks}"
RESULTS_BASE="${RESULTS_BASE:-/shared/healthcheck-results/sweeps}"
NODES_PER_SWEEP="${NODES_PER_SWEEP:-10}"
PARTITION="${PARTITION:-gpu}"

SWEEP_ID="sweep-$(date +%Y%m%d-%H%M%S)"
SWEEP_RESULTS="${RESULTS_BASE}/${SWEEP_ID}"

echo "[${SWEEP_ID}] Starting rolling GPU health check sweep"
echo "[${SWEEP_ID}] Max nodes per sweep: ${NODES_PER_SWEEP}"

# ─── Find idle GPU nodes ────────────────────────────────────────────────────
IDLE_NODES=$(sinfo -p "${PARTITION}" -t idle -N -h -o "%N" 2>/dev/null | head -n "${NODES_PER_SWEEP}")

if [[ -z "${IDLE_NODES}" ]]; then
    echo "[${SWEEP_ID}] No idle nodes found in partition '${PARTITION}' -- skipping sweep"
    exit 0
fi

NODE_COUNT=$(echo "${IDLE_NODES}" | wc -l | tr -d ' ')
echo "[${SWEEP_ID}] Found ${NODE_COUNT} idle node(s) to check"

mkdir -p "${SWEEP_RESULTS}"

# ─── Submit health check jobs ───────────────────────────────────────────────
SUBMITTED=0
FAILED=0

while IFS= read -r node; do
    [[ -z "${node}" ]] && continue

    echo "[${SWEEP_ID}] Submitting lightweight check for node: ${node}"

    JOB_ID=$(sbatch \
        --job-name="gpu-sweep-${node}" \
        --nodelist="${node}" \
        --ntasks=1 \
        --time=00:20:00 \
        --output="${SWEEP_RESULTS}/${node}-%j.out" \
        --error="${SWEEP_RESULTS}/${node}-%j.err" \
        --export=ALL,HEALTHCHECK_DIR=${HEALTHCHECK_DIR},RESULTS_BASE=${SWEEP_RESULTS} \
        "${HEALTHCHECK_DIR}/slurm/sbatch-lightweight.sh" \
        2>&1 | grep -oE "[0-9]+" || true)

    if [[ -n "${JOB_ID}" ]]; then
        echo "[${SWEEP_ID}]   Submitted job ${JOB_ID} for ${node}"
        SUBMITTED=$((SUBMITTED + 1))
    else
        echo "[${SWEEP_ID}]   Failed to submit job for ${node}"
        FAILED=$((FAILED + 1))
    fi
done <<< "${IDLE_NODES}"

echo "[${SWEEP_ID}] Sweep complete: ${SUBMITTED} submitted, ${FAILED} failed"
echo "[${SWEEP_ID}] Results will be in: ${SWEEP_RESULTS}"

# Write sweep metadata
cat > "${SWEEP_RESULTS}/sweep-metadata.json" <<ENDJSON
{
  "sweep_id": "${SWEEP_ID}",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "partition": "${PARTITION}",
  "nodes_checked": ${SUBMITTED},
  "nodes_failed_submit": ${FAILED},
  "max_per_sweep": ${NODES_PER_SWEEP}
}
ENDJSON
