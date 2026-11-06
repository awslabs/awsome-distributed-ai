#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Facilitator installs this as a compute-node Prolog. See README.md.
set -euo pipefail
export PATH="/opt/amazon/efa/bin:${PATH:-/usr/bin:/bin}"
suite=/opt/aim344-healthcheck/validation/gpu-cluster-healthcheck
# Match the root-owned assignment used by maintenance, not job-supplied counts.
unset HEALTHCHECK_EXPECTED_EFA_COUNT INSTANCE_TYPE
if [[ -e /etc/aim344-maintenance.json || -L /etc/aim344-maintenance.json ]]; then
    assigned_efa=$(python3 - <<'PY'
import json, os, stat
path = '/etc/aim344-maintenance.json'
fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
with os.fdopen(fd) as source:
    info = os.fstat(source.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o077:
        raise SystemExit('Untrusted maintenance assignment')
    config = json.load(source)
if 'expected_efa_devices' in config:
    count = config['expected_efa_devices']
    if type(count) is not int or not 1 <= count <= 32:
        raise SystemExit('Invalid assigned EFA count')
    print(count)
PY
    )
    if [[ -n $assigned_efa ]]; then
        export HEALTHCHECK_EXPECTED_EFA_COUNT=$assigned_efa
    fi
fi
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
