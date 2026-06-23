#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Short Megatron-Core MoE run on GB200 (Slurm). Keep EP within the 72-GPU NVLink domain.
#SBATCH --job-name=gb200-moe-train
#SBATCH --nodes=18
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output %x_%j.out
#SBATCH --error  %x_%j.err
set -euo pipefail
: "${APPS_PATH:=/fsx}"
: "${IMAGE:=$APPS_PATH/gb200-moe-arm64.sqsh}"
: "${BACKEND:=hybridep}"
: "${EP:=8}"
export BACKEND EP
srun --container-image "$IMAGE" --mpi=pmix --cpu-bind=none \
  bash -lc 'cd /opt/DeepEP && BACKEND='"$BACKEND"' EP='"$EP"' ./train-moe.sh'
