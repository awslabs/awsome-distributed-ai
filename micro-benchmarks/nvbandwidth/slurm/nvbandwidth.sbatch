#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# nvbandwidth fabric test for P6e-GB200 on Slurm. Validates NVLink-C2C, intra-domain
# GPU<->GPU bisection, and the IMEX cross-instance path within one 72-GPU NVLink domain.

#SBATCH --job-name=nvbandwidth-gb200
#SBATCH --nodes=18                 # one full 72-GPU NVLink domain (1 UltraServer)
#SBATCH --ntasks-per-node=4        # 4 GB200 GPUs per p6e-gb200.36xlarge
#SBATCH --gpus-per-node=4
#SBATCH --exclusive                # IMEX / NVLink-domain semantics require exclusive nodes
#SBATCH --wait-all-nodes=1
#SBATCH --output %x_%j.out
#SBATCH --error  %x_%j.err

set -euo pipefail

: "${APPS_PATH:=/fsx}"
: "${IMAGE:=$APPS_PATH/nvbandwidth-arm64.sqsh}"
: "${NVBW_MIN_GBPS:=600}"          # per-direction floor; calibrate to your measured C2C baseline

# --- IMEX preflight: the cross-instance NVLink domain must be UP before any fabric test ---
echo "== IMEX domain preflight =="
srun -N "$SLURM_NNODES" --ntasks-per-node=1 bash -c \
  'nvidia-imex-ctl -N | grep -q "Domain State: UP" || { echo "IMEX NOT UP on $(hostname)"; exit 1; }'
echo "IMEX domain UP on all $SLURM_NNODES nodes."

declare -a ARGS=( --container-image "$IMAGE" )

# Grace<->Blackwell C2C (host<->device) + intra-domain device<->device, multi-node.
srun "${ARGS[@]}" --mpi=pmix --cpu-bind=none \
  nvbandwidth -t multinode_device_to_device_memcpy_read_ce

srun "${ARGS[@]}" --mpi=pmix --cpu-bind=none \
  nvbandwidth -t host_to_device_memcpy_ce -t device_to_host_memcpy_ce   # C2C path

echo "Floor check: NVBW_MIN_GBPS=${NVBW_MIN_GBPS} (enforced by test_nvbandwidth.py in CI)"
