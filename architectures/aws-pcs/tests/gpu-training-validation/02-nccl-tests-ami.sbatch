#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Stage 2 (Approach 1 - AMI native, FAST): 2-node NCCL all_reduce_perf over EFA
# using the all_reduce_perf binary PRE-BUILT into the PCS DLAMI, plus the bundled
# aws-ofi-nccl + EFA libraries. No container build, no Pyxis required.
# Reference: https://catalog.workshops.aws/ml-on-pcs/en-US/05-test-gpu/06-nccl-test
#
# Adapted from micro-benchmarks/nccl-tests/slurm/nccl-tests-ami.sbatch.

#SBATCH --job-name=nccl-allreduce-ami
#SBATCH --partition=gpu-p5
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8        # 8 GPUs per p5.48xlarge
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1
#SBATCH --output=02-nccl-tests-ami_%j.out
#SBATCH --error=02-nccl-tests-ami_%j.out

set -euxo pipefail

# Pre-built binary shipped on the PCS DLAMI. cuda-13.2 is the default toolkit on
# this AMI; override via the first arg if needed.
ALL_REDUCE_BINARY=${1:-/usr/local/cuda-13.2/efa/test-cuda-13.2/all_reduce_perf}
ADDITIONAL_LD_LIBRARY_PATH=${2:-/usr/local/cuda-13.2/lib}

[[ -x "${ALL_REDUCE_BINARY}" ]] || { echo "Missing ${ALL_REDUCE_BINARY} on the AMI" >&2; exit 1; }

# Hostname -> instance-id mapping (one rank per node)
mpirun -N 1 bash -c 'echo $(hostname): $(cat /sys/devices/virtual/dmi/id/board_asset_tag | tr -d " ")'

# all_reduce across all 16 GPUs, 8 B to 16 GB
mpirun -n $((8 * SLURM_JOB_NUM_NODES)) -N 8 \
    -x FI_PROVIDER=efa \
    -x FI_EFA_FORK_SAFE=1 \
    -x LD_LIBRARY_PATH="${ADDITIONAL_LD_LIBRARY_PATH}:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/amazon/ofi-nccl/lib:/usr/local/lib:/usr/lib:${LD_LIBRARY_PATH:-}" \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=^docker,lo,veth \
    --bind-to none "${ALL_REDUCE_BINARY}" -b 8 -e 16G -f 2 -g 1 -c 1 -n 100
