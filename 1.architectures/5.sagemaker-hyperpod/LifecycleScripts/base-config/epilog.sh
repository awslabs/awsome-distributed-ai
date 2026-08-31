#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Slurm Epilog: removes GPU-to-Job mapping files written by prolog.sh.
# Called by Slurm when a job ends (normally or due to failure/cancellation).

LOG_FILE="/tmp/slurm_epilog_${SLURM_JOB_ID}.log"

# Must match the host path used in prolog.sh and mounted into the DCGM
# Exporter container.
# See install_dcgm_exporter.sh: -v "$JOB_ID_MAP_DIR:/etc/dcgm-exporter/hpc"
JOB_ID_MAP_DIR="/run/slurm/dcgm_job_mapping"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"; }

log "=== Slurm Epilog - Job $SLURM_JOB_ID ==="
log "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
log "JOB_ID_MAP_DIR: $JOB_ID_MAP_DIR"

if [ -n "$SLURM_JOB_GPUS" ]; then
    IFS=',' read -ra GPUS <<< "$SLURM_JOB_GPUS"
    for GPU in "${GPUS[@]}"; do
        FILE="$JOB_ID_MAP_DIR/$GPU"
        if [ -f "$FILE" ]; then
            # Remove only the line belonging to this job so that other jobs
            # sharing the same GPU (e.g. MIG / GRES sharing) are unaffected.
            sed -i "/^${SLURM_JOB_ID}$/d" "$FILE"
            # Clean up the file once no jobs remain mapped to this GPU.
            if [ ! -s "$FILE" ]; then
                rm -f "$FILE"
                log "Removed mapping file $FILE (GPU $GPU now idle)"
            else
                log "Removed Job $SLURM_JOB_ID from $FILE (other jobs still active on GPU $GPU)"
            fi
        else
            log "WARNING: Mapping file $FILE not found for GPU $GPU — already cleaned up?"
        fi
    done
    log "GPU mapping cleanup complete"
else
    log "WARNING: SLURM_JOB_GPUS is empty — no cleanup performed (non-GPU job?)"
fi

log "Mapping directory contents: $(ls "$JOB_ID_MAP_DIR" 2>&1)"

if [ -d /etc/otel ]; then
    log "Removing the otel collector target.json"
    cat > /etc/otel/targets.json <<EOF
[{"targets": ["localhost:9100"], "labels": {}}, {"targets": ["localhost:9109"], "labels": {}}]
EOF
else
    log "Skipping otel target cleanup — /etc/otel not found (observability not installed)"
fi