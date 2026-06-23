#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# NCCL test for P6e-GB200 on Slurm (HyperPod-Slurm / ParallelCluster).
#
# Sizing:
#   * INTRA one UltraServer (one 72-GPU NVLink domain): --nodes up to 18 (=72 GPUs).
#     The collective stays on NVSwitch (NVLS / NVLink SHARP); EFA is not on the path.
#   * CROSS UltraServers (>72 GPUs over EFA, no in-network SHARP): --nodes 36 (=144).
#     Requires two UltraServers; authored-to-spec until validated at scale.
#
# Each p6e-gb200.36xlarge contributes 4 GB200 GPUs, so --ntasks-per-node=4.

#SBATCH --job-name=nccl-gb200-all_reduce
#SBATCH --nodes=18                 # 18 = one full 72-GPU NVLink domain; set 36 for cross-domain
#SBATCH --ntasks-per-node=4        # 4 GB200 GPUs per p6e-gb200.36xlarge
#SBATCH --gpus-per-node=4
#SBATCH --exclusive                # required: IMEX / NVLink-domain semantics need exclusive nodes
#SBATCH --wait-all-nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --output %x_%j.out
#SBATCH --error  %x_%j.err

set -euo pipefail

: "${APPS_PATH:=/fsx}"
: "${NCCL_TESTS_PATH:=/opt/nccl-tests/build}"
: "${IMAGE:=$APPS_PATH/nccl-tests-arm64.sqsh}"

# Canonical GB200 NCCL + EFA environment (NVLS intra-domain, NVLSTree over EFA cross-domain).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/gb200-env.sh"

declare -a ARGS=( --container-image "$IMAGE" )

# Print host + UltraServer/board asset tag so you can confirm the node-to-clique mapping.
srun -N "$SLURM_NNODES" --ntasks-per-node=1 bash -c \
  'echo "$(hostname): $(cat /sys/devices/virtual/dmi/id/board_asset_tag 2>/dev/null | tr -d " ")"'

# all_reduce: -c 1 turns on the correctness check (the #wrong column must be 0).
srun "${ARGS[@]}" --mpi=pmix --cpu-bind=none \
  "$NCCL_TESTS_PATH/all_reduce_perf" -b 8 -e 16G -f 2 -g 1 -c 1 -n 100
