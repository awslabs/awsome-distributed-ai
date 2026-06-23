#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Mandatory GB200 NCCL correctness gate (Slurm). Run before production on a domain.
#SBATCH --job-name=gb200-correctness-gate
#SBATCH --nodes=18                 # one 72-GPU NVLink domain
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1
#SBATCH --output %x_%j.out
#SBATCH --error  %x_%j.err

set -euo pipefail
: "${APPS_PATH:=/fsx}"
: "${IMAGE:=$APPS_PATH/nccl-tests-arm64.sqsh}"
: "${NCCL_TESTS_PATH:=/opt/nccl-tests/build}"

# Canonical GB200 NCCL+EFA env (NVLS intra-domain).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../micro-benchmarks/nccl-tests/gb200-env.sh"

# all_reduce with -c 1, piped into the correctness gate (fails on any #wrong != 0).
srun --container-image "$IMAGE" --mpi=pmix --cpu-bind=none \
  "$NCCL_TESTS_PATH/all_reduce_perf" -b 8 -e 16G -f 2 -g 1 -c 1 -n 100 \
  | bash "${SCRIPT_DIR}/../nccl-correctness-gate.sh"
