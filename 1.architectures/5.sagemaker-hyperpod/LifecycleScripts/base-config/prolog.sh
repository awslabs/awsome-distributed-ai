#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Slurm Prolog: writes GPU-to-Job mapping files for DCGM Exporter.
# DCGM Exporter reads this directory (--hpc-job-mapping-dir) to associate
# GPU metrics with Slurm job IDs.
#   - Filename : GPU index (as assigned by Slurm via SLURM_JOB_GPUS)
#   - Content  : Slurm Job ID

LOG_FILE="/tmp/slurm_prolog_${SLURM_JOB_ID}.log"

# Must match the host path mounted into the DCGM Exporter container.
# See install_dcgm_exporter.sh: -v "$JOB_ID_MAP_DIR:/etc/dcgm-exporter/hpc"
JOB_ID_MAP_DIR="/run/slurm/dcgm_job_mapping"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"; }

log "=== Slurm Prolog - Job $SLURM_JOB_ID ==="
log "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
log "JOB_ID_MAP_DIR: $JOB_ID_MAP_DIR"

# Ensure the mapping directory exists
if ! mkdir -p "$JOB_ID_MAP_DIR"; then
    log "ERROR: Failed to create $JOB_ID_MAP_DIR — aborting GPU mapping"
    exit 1
fi

if [ -n "$SLURM_JOB_GPUS" ]; then
    IFS=',' read -ra GPUS <<< "$SLURM_JOB_GPUS"
    for GPU in "${GPUS[@]}"; do
        # Append the job ID so that if multiple jobs share a GPU (e.g. MIG or
        # GRES sharing), all job IDs are preserved.  Each job occupies exactly
        # one line.  The epilog removes only the line for the finishing job.
        echo "$SLURM_JOB_ID" >> "$JOB_ID_MAP_DIR/$GPU"
        log "Wrote Job $SLURM_JOB_ID -> $JOB_ID_MAP_DIR/$GPU"
    done
    log "GPU mapping complete"
else
    log "WARNING: SLURM_JOB_GPUS is empty — no mapping files created (non-GPU job?)"
fi

log "Mapping directory contents: $(ls "$JOB_ID_MAP_DIR" 2>&1)"

if [ -d /etc/otel ]; then
    log "Updating the otel collector target.json"
    cat > /etc/otel/targets.json <<EOF
[{"targets": ["localhost:9100"], "labels": {"slurm_job_id": "${SLURM_JOB_ID}"}},{"targets": ["localhost:9109"], "labels": {"slurm_job_id": "${SLURM_JOB_ID}"}}]
EOF
else
    log "Skipping otel target update — /etc/otel not found (observability not installed)"
fi
