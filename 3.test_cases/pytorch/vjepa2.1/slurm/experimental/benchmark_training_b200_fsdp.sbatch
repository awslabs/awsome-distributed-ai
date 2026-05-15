#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# FSDP benchmark script for V-JEPA 2.1 on SageMaker HyperPod with B200 GPUs.
#
# Uses FSDP (SHARD_GRAD_OP / ZeRO-2) instead of DDP to shard gradients and
# optimizer states, saving ~15 GB/GPU.  This enables disabling activation
# checkpointing for higher throughput.
#
# See benchmark_training_b200.sbatch for prerequisites and setup instructions.

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --job-name=vjepa21-fsdp
#SBATCH --output=logs/vjepa21_fsdp/%A.out
#SBATCH --error=logs/vjepa21_fsdp/%A.err
#SBATCH --exclusive
#SBATCH --partition=b200
#SBATCH --account=root
#SBATCH --time=01:00:00

set -euo pipefail

###########################
###### User Variables #####
###########################

VJEPA21_DIR=/fsx/${USER}/vjepa2.1
VJEPA2_CODE=/fsx/${USER}/vjepa2_code
VJEPA_DEPS=/fsx/${USER}/vjepa_deps

###########################
## Container Variables ####
###########################

export CONTAINER_IMAGE=/fsx/${USER}/nemo-efa-nccl29.sqsh
export FSX_MOUNT=/fsx:/fsx
export CODE_MOUNT=${VJEPA2_CODE}:/workspace

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

# V-JEPA 2.1 needs /workspace in PYTHONPATH for the code overlay
export PYTHONPATH=/workspace:${VJEPA_DEPS}:${VJEPA2_CODE}:${PYTHONPATH:-}

###########################
## Container Args #########
###########################

declare -a ARGS=(
    --container-image $CONTAINER_IMAGE
    --container-mounts "${FSX_MOUNT},${CODE_MOUNT}"
    --container-workdir /workspace
    --no-container-mount-home
    --mpi=pmix
)

###########################
## Training Args ##########
###########################

# Uses run_train_fsdp.py instead of run_train.py
declare -a TRAINING_ARGS=(
    --fname ${VJEPA21_DIR}/configs/experimental/benchmark-vitg-8nodes-fsdp.yaml
)

###########################
## HyperPod Auto-Resume ###
###########################

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi

mkdir -p logs/vjepa21_fsdp

srun ${AUTO_RESUME} -l "${ARGS[@]}" python ${VJEPA21_DIR}/scripts/experimental/run_train_fsdp.py "${TRAINING_ARGS[@]}"
