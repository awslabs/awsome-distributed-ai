#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# FP4/FP8 Transformer Engine GEMM microbenchmark on one P6e-GB200 node (4 GB200 GPUs).

#SBATCH --job-name=te-gemm-gb200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output %x_%j.out
#SBATCH --error  %x_%j.err

set -euo pipefail
: "${APPS_PATH:=/fsx}"
: "${IMAGE:=$APPS_PATH/transformer-engine-gemm-arm64.sqsh}"

# Single-node, GPU-only -- no EFA/NCCL env needed. One rank drives all 4 GPUs in turn.
srun --container-image "$IMAGE" --cpu-bind=none bash -lc '
  cd /opt/te-gemm && ./run_gemm_bench.sh && python3 roofline.py results.json
'
