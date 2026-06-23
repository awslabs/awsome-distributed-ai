#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# FP8/FP4 Megatron training on P6e-GB200 (Slurm). Default: one UltraServer (72 GPUs).
#SBATCH --job-name=gb200-megatron-fp8
#SBATCH --nodes=18                 # 72 GPUs; set 36 for two UltraServers (DP/PP over EFA)
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1
#SBATCH --output %x_%j.out
#SBATCH --error  %x_%j.err

set -euo pipefail
: "${APPS_PATH:=/fsx}"
: "${IMAGE:=$APPS_PATH/gb200-megatron-fp8-arm64.sqsh}"
: "${PRECISION:=fp8-mxfp8}"
: "${MODEL:=llama3-8b}"

export PRECISION MODEL

srun --container-image "$IMAGE" --mpi=pmix --cpu-bind=none \
  bash -lc 'cd /opt/Megatron-LM && PRECISION='"$PRECISION"' MODEL='"$MODEL"' ./train.sh'
