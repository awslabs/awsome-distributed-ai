#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Slurm Prolog for the lab's partition: run Checks 0 and 2 of the health suite on
# the node before each job. A failure drains the node and requeues the job.
set -euo pipefail
[[ ${SLURM_JOB_USER:-} != root ]] || exit 0
export PATH="/opt/amazon/efa/bin:${PATH:-/usr/bin:/bin}"
suite=/opt/aim344-healthcheck/validation/gpu-cluster-healthcheck
# lab.env carries HEALTHCHECK_EXPECTED_EFA_COUNT, the node's EFA device count.
lab_env=/opt/aim344/device-recovery/companion/lab.env
# shellcheck source=/dev/null
[[ ! -r $lab_env ]] || source "$lab_env"
RESULTS_DIR=$(mktemp -d "/var/log/aim344-prolog-job-${SLURM_JOB_ID:-unknown}.XXXXXX")
export RESULTS_DIR
for check in 0-nvidia-smi-check 2-efa-enumeration; do
    if ! bash "$suite/checks/$check.sh" > "$RESULTS_DIR/$check.log" 2>&1; then
        logger -t aim344-prolog -f "$RESULTS_DIR/$check.log"
        logger -t aim344-prolog -- "Health check failed: check=$check job=${SLURM_JOB_ID:-unknown}."
        exit 1
    fi
done
logger -t aim344-prolog -- "Health gate passed: job=${SLURM_JOB_ID:-unknown} node=$(hostname)."
