#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Facilitator installs this as a compute-node Prolog. See README.md.
set -euo pipefail
suite=/opt/aim344-healthcheck/validation/gpu-cluster-healthcheck
marker=/run/aim344-unhealthy
log() { logger -t aim344-prolog -- "$*"; }
if [[ -f $marker && ! -L $marker && $(stat -c %u "$marker") == 0 ]]; then
    read -r tag target_user < "$marker"
    if [[ $tag == AIM344 && $target_user == "${SLURM_JOB_USER:-}" ]]; then
        log "Synthetic unhealthy verdict: job=${SLURM_JOB_ID:-unknown} node=$(hostname)."
        exit 1
    fi
fi
install -d -m 0700 /var/log/aim344-prolog
RESULTS_DIR=$(mktemp -d "/var/log/aim344-prolog/job-${SLURM_JOB_ID:-unknown}.XXXXXX")
export RESULTS_DIR
for check in 0-nvidia-smi-check 2-efa-enumeration; do
    if ! bash "$suite/checks/$check.sh" > "$RESULTS_DIR/$check.log" 2>&1; then
        logger -t aim344-prolog -f "$RESULTS_DIR/$check.log"
        log "Health check failed: check=$check job=${SLURM_JOB_ID:-unknown}."
        exit 1
    fi
    logger -t aim344-prolog -f "$RESULTS_DIR/$check.log"
done
log "Health gate passed: job=${SLURM_JOB_ID:-unknown} node=$(hostname)."
