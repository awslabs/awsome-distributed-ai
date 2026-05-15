#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# nsys profiling wrapper for rank-selective profiling.
#
# Only rank 0 (SLURM_PROCID=0) is profiled with nsys to avoid generating
# large output files from every GPU.  All other ranks run the command directly.
#
# Environment variables:
#   NSYS_OUTPUT  - Output path for the .nsys-rep file (without extension).
#                  Default: /tmp/nsys_profile_rank0
#
# Usage (called from sbatch via srun):
#   srun ... bash scripts/nsys_wrapper.sh python scripts/run_train.py --fname config.yaml

set -euo pipefail

NSYS_BIN="${NSYS_BIN:-/usr/local/cuda/bin/nsys}"
NSYS_OUTPUT="${NSYS_OUTPUT:-/tmp/nsys_profile_rank0}"

if [ "${SLURM_PROCID}" = "0" ]; then
    echo "[nsys_wrapper] Rank 0: profiling with nsys -> ${NSYS_OUTPUT}.nsys-rep"
    mkdir -p "$(dirname "${NSYS_OUTPUT}")"
    exec "${NSYS_BIN}" profile \
        -o "${NSYS_OUTPUT}" \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --cuda-memory-usage=true \
        --force-overwrite=true \
        "$@"
else
    exec "$@"
fi
