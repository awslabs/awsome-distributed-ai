#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Run this from the unaffected coordinator during the facilitator's device round.
set -euo pipefail
LAB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$LAB_DIR/common.sh"
# Unlike the storage probes, device cleanup may outlive the watchdog. Slurm,
# not a SIGKILL of the evidence-producing launcher, owns the outer deadline.
: "${SLURM_JOB_ID:?Use your dedicated allocation with --time=00:06:00}"
limit=$(scontrol show job "$SLURM_JOB_ID" -o)
[[ $limit =~ TimeLimit=(00:0[1-9]:[0-5][0-9]|00:10:00)([[:space:]]|$) ]] || {
    printf 'Use a device allocation with a walltime of 1-10 minutes (recommended 00:06:00).\n' >&2
    exit 2
}
prepare_slurm
run_torch device device --duration-seconds=120
