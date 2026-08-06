#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Validates that the vllm-deepep-efa image actually moves expert traffic over
# EFA, on real GPU nodes with the host driver injected -- the one thing the
# image's own build-time smoke tests cannot check.
#
# Launch convention matches
# micro-benchmarks/expert-parallelism/deepep-benchmark/slurm/test_internode.sbatch:
# ONE task per node. DeepEP's test spawns its own 8 ranks per node via
# torch.multiprocessing, so --ntasks-per-node 8 makes each task build its own
# 8-rank group and the run fails on either
# 'num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode' or EADDRINUSE.
#
#   sbatch validate_2node.sbatch /fsx/$USER/containers/vllm-deepep-efa.sqsh

#SBATCH --job-name=vllm-deepep-validate
#SBATCH --nodes=2
#SBATCH --ntasks-per-node 1
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1
#SBATCH --time=00:30:00

IMG="${1:-/fsx/${USER}/containers/vllm-deepep-efa.sqsh}"

# The four NVSHMEM/libfabric vars are deliberately NOT exported here: they are
# baked into the image ENV, and part of what this job checks is that
# `enroot import` preserved them. Section A fails loudly if they did not.
export MASTER_ADDR=$(scontrol show hostname "${SLURM_NODELIST}" | head -n 1)
export WORLD_SIZE=$SLURM_NNODES

echo "=== A: transport vars survived enroot import into the squashfs ==="
srun -N1 -n1 --container-image "${IMG}" bash -c '
    for v in FI_PROVIDER NVSHMEM_REMOTE_TRANSPORT NVSHMEM_LIBFABRIC_PROVIDER \
             NVSHMEM_NETDEVS_POLICY; do
        eval "val=\$$v"
        if [ -z "$val" ]; then echo "MISSING: $v"; exit 1; fi
        echo "  $v=$val"
    done'

echo ""
echo "=== B: driver and EFA visible inside the container ==="
srun -N1 -n1 --container-image "${IMG}" bash -c '
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
    echo "  EFA endpoints: $(fi_info -p efa 2>/dev/null | grep -c provider)"'

echo ""
echo "=== C: DeepEP internode dispatch/combine over EFA (2 nodes x 8 GPUs) ==="
srun --mpi=pmix --cpu-bind=none --container-image "${IMG}" \
    bash -c 'RANK=$SLURM_PROCID python3 /opt/amazon/deepep/tests/test_internode.py'
