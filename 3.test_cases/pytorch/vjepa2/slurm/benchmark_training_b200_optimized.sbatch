#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Optimized benchmark script for V-JEPA 2 on SageMaker HyperPod with B200 GPUs.
#
# Uses the optimized config with torch.compile enabled and activation
# checkpointing disabled for maximum throughput.  This increases GPU memory
# usage to ~95 GB/GPU (vs ~33 GB baseline) but provides ~23% speedup.
#
# See benchmark_training_b200.sbatch for prerequisites and setup instructions.

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --job-name=vjepa2-bench-opt
#SBATCH --output=logs/vjepa2_benchmark_opt/%A.out
#SBATCH --error=logs/vjepa2_benchmark_opt/%A.err
#SBATCH --exclusive
#SBATCH --partition=b200
#SBATCH --account=root
#SBATCH --time=01:00:00

set -euo pipefail

###########################
###### User Variables #####
###########################

VJEPA2_DIR=/fsx/${USER}/vjepa2
VJEPA2_CODE=/fsx/${USER}/vjepa2_code
VJEPA_DEPS=/fsx/${USER}/vjepa_deps

###########################
## Container Variables ####
###########################

export CONTAINER_IMAGE=/fsx/${USER}/nemo-efa-nccl29.sqsh
export FSX_MOUNT=/fsx:/fsx

###########################
## Environment Variables ##
###########################

export NCCL_DEBUG=INFO
export FI_PROVIDER=efa
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export NCCL_SOCKET_IFNAME=^docker,lo,veth

# NCCL performance tuning for EFA on HyperPod B200
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_NVLS_ENABLE=0
export NCCL_GRAPH_REGISTER=0

# Blackwell-specific
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# PyTorch memory allocator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Add V-JEPA 2 deps and source code to Python path
export PYTHONPATH=${VJEPA_DEPS}:${VJEPA2_CODE}:${PYTHONPATH:-}

###########################
## Container Args #########
###########################

declare -a ARGS=(
    --container-image $CONTAINER_IMAGE
    --container-mounts $FSX_MOUNT
    --container-workdir ${VJEPA2_CODE}
    --no-container-mount-home
    --mpi=pmix
)

###########################
## Training Args ##########
###########################

declare -a TRAINING_ARGS=(
    --fname ${VJEPA2_DIR}/configs/benchmark-vitg-8nodes-optimized.yaml
)

###########################
## HyperPod Auto-Resume ###
###########################

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi

mkdir -p logs/vjepa2_benchmark_opt

srun ${AUTO_RESUME} -l "${ARGS[@]}" python ${VJEPA2_DIR}/scripts/run_train.py "${TRAINING_ARGS[@]}"
