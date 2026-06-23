#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# DeepEP dispatch/combine bandwidth on GB200. SCOPE=intra (1 UltraServer) or cross (2).
#SBATCH --job-name=gb200-ep-bench
#SBATCH --nodes=18                 # SCOPE=intra: 18 (72 GPUs). SCOPE=cross: submit with --nodes=36
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output %x_%j.out
#SBATCH --error  %x_%j.err
set -euo pipefail
: "${APPS_PATH:=/fsx}"
: "${IMAGE:=$APPS_PATH/gb200-moe-arm64.sqsh}"
: "${SCOPE:=intra}"
export SCOPE
srun --container-image "$IMAGE" --mpi=pmix --cpu-bind=none \
  bash -lc 'cd /opt/DeepEP && SCOPE='"$SCOPE"' ./ep-bench.sh'
