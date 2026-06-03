#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Stage 2: 2-node NCCL all_reduce_perf over EFA, via Pyxis/Enroot using the
# prebuilt published image from public ECR (no local build):
#   public.ecr.aws/hpc-cloud/nccl-tests
# Import it first on the login node with 02-import-nccl-image.sh (writes
# /fsx/nccl-tests.sqsh). Adjust --partition to your GPU queue
# (gpu-p5 / gpu-p6b200 / gpu-p6b300).
# Reference: https://catalog.workshops.aws/ml-on-pcs/en-US/05-test-gpu/06-nccl-test

#SBATCH --job-name=nccl-all_reduce_perf
#SBATCH --partition=gpu-p6b200
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8        # 8 GPUs per node
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1
#SBATCH --output=02-nccl-tests_%j.out
#SBATCH --error=02-nccl-tests_%j.out

set -euxo pipefail

: "${IMAGE:=/fsx/nccl-tests.sqsh}"
: "${NCCL_TESTS_PATH:=/opt/nccl-tests/build}"

[[ -f "${IMAGE}" ]] || { echo "Missing ${IMAGE}; run 02-import-nccl-image.sh first." >&2; exit 1; }

## EFA / libfabric
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
export NCCL_DEBUG=INFO

declare -a ARGS=(
    --container-image "${IMAGE}"
)

# Hostname -> instance-id mapping (one task per node)
srun -N "${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
     bash -c 'echo $(hostname): $(cat /sys/devices/virtual/dmi/id/board_asset_tag | tr -d " ")'

# all_reduce across all 16 GPUs, 8 B to 16 GB
srun "${ARGS[@]}" --mpi=pmix --cpu-bind=none \
     "${NCCL_TESTS_PATH}/all_reduce_perf" -b 8 -e 16G -f 2 -g 1 -c 1 -n 100
