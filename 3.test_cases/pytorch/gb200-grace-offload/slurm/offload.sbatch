#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Grace unified-memory offload training on one P6e-GB200 node (4 GB200 GPUs + Grace).
#SBATCH --job-name=gb200-grace-offload
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output %x_%j.out
#SBATCH --error  %x_%j.err
set -euo pipefail
: "${APPS_PATH:=/fsx}"
: "${IMAGE:=$APPS_PATH/gb200-grace-offload-arm64.sqsh}"
: "${VARIANT:=superoffload}"
export VARIANT
# --container-privileged enables MPAM/resctrl bandwidth partitioning on bare-metal/Slurm.
srun --container-image "$IMAGE" --container-privileged --cpu-bind=none \
  bash -lc 'cd /opt/grace-offload && VARIANT='"$VARIANT"' ./run.sh'
