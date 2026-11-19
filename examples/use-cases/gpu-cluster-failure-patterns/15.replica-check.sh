#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Replica check: every rank computes the same matrix product and the ranks compare
# bit-exact checksums each step, beside the usual all-reduce check. Extra arguments
# reach workload.py; to alter one GPU's result in software, pass
#   --corrupt-rank R [--corrupt-step S] [--corrupt-bit B] [--corrupt-recurring]
# The job raises nothing: read the "AIM344 replica check:" lines in the job log.
set -euo pipefail
LAB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$LAB_DIR/common.sh"
prepare_slurm
export FI_EFA_USE_HUGE_PAGE=${FI_EFA_USE_HUGE_PAGE:-0}
run_torch replica-check device --replica-check --duration-seconds=60 "$@"
