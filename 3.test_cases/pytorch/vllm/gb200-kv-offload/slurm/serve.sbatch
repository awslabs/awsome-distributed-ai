#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# KV-offload-to-Grace serving on one P6e-GB200 node (Slurm).
#SBATCH --job-name=gb200-kv-offload
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output %x_%j.out
#SBATCH --error  %x_%j.err
set -euo pipefail
: "${APPS_PATH:=/fsx}"
: "${IMAGE:=$APPS_PATH/gb200-kv-offload-arm64.sqsh}"
: "${MODEL:=deepseek-ai/DeepSeek-V3}"
: "${CPU_OFFLOAD_GB:=320}"
export MODEL CPU_OFFLOAD_GB
srun --container-image "$IMAGE" --cpu-bind=none bash -lc 'cd /opt/kv-offload && ./serve.sh'
