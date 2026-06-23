#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# slime GRPO RL post-training on P6e-GB200 (Slurm). Ray pinned inside the NVLink domain.
#SBATCH --job-name=gb200-slime-grpo
#SBATCH --nodes=18                 # one 72-GPU NVLink domain
#SBATCH --ntasks-per-node=1        # one Ray worker per node; it manages the 4 local GPUs
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1
#SBATCH --output %x_%j.out
#SBATCH --error  %x_%j.err
set -euo pipefail
: "${APPS_PATH:=/fsx}"
: "${IMAGE:=$APPS_PATH/gb200-slime-arm64.sqsh}"
: "${MODEL:=Qwen/Qwen3-30B-A3B}"
export MODEL

# Start Ray head on the first node, workers on the rest, all within the clique, then GRPO.
srun --container-image "$IMAGE" --cpu-bind=none bash -lc '
  cd /opt/slime && MODEL='"$MODEL"' ./run-grpo.sh
'
