#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Run this from the unaffected coordinator during the facilitator's device round.
set -euo pipefail
[[ $# -eq 0 ]] || { echo "Usage: bash ${0##*/} (it takes no arguments)" >&2; exit 2; }
LAB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$LAB_DIR/common.sh"
# Unlike the storage probes, device cleanup may outlive the watchdog. Slurm,
# not a SIGKILL of the evidence-producing launcher, owns the outer deadline:
# run_torch gives the device step its own Slurm time limit.
: "${SLURM_JOB_ID:?Run this inside your dedicated aim344-device allocation}"
prepare_slurm
run_torch device device --duration-seconds=120
