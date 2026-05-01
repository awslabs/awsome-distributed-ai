#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# nsys profiling script for V-JEPA 2.1 on SageMaker HyperPod with B200 GPUs.
#
# Profiles rank 0 only using nsys_wrapper.sh.  All other ranks run
# un-instrumented to avoid large output files and profiling overhead.
#
# Environment variables (set before sbatch or use --export):
#   NSYS_PROFILE_DIR - Subfolder name under nsys/ for this run.
#                      Default: "baseline".  Examples: phase2_noscaler, phase4_fsdp
#   CONFIG           - Path to YAML config.  Default: baseline config.
#
# Usage:
#   # Baseline profile
#   sbatch slurm/nsys_profile_b200.sbatch
#
#   # Profile after Phase 4 (FSDP)
#   NSYS_PROFILE_DIR=phase4_fsdp \
#   CONFIG=/fsx/${USER}/vjepa2.1/configs/benchmark-vitg-8nodes-fsdp.yaml \
#       sbatch slurm/nsys_profile_b200.sbatch

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --job-name=vjepa21-nsys
#SBATCH --output=logs/vjepa21_nsys/%A.out
#SBATCH --error=logs/vjepa21_nsys/%A.err
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

# Profiling output directory — override with NSYS_PROFILE_DIR env var
NSYS_PROFILE_DIR="${NSYS_PROFILE_DIR:-baseline}"
export NSYS_OUTPUT=${VJEPA21_DIR}/nsys/${NSYS_PROFILE_DIR}/rank0

# Config — override with CONFIG env var
CONFIG="${CONFIG:-${VJEPA21_DIR}/configs/benchmark-vitg-8nodes.yaml}"

###########################
## Container Variables ####
###########################

export CONTAINER_IMAGE=/fsx/${USER}/nemo-efa-nccl29.sqsh
export FSX_MOUNT=/fsx:/fsx

# V-JEPA 2.1 code overlay: mount code at /workspace so it takes
# priority in the NeMo container's PYTHONPATH (which includes /workspace).
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

declare -a TRAINING_ARGS=(
    --fname ${CONFIG}
)

###########################
## HyperPod Auto-Resume ###
###########################

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi

echo "=== nsys profiling ==="
echo "Profile dir: ${NSYS_PROFILE_DIR}"
echo "Output:      ${NSYS_OUTPUT}.nsys-rep"
echo "Config:      ${CONFIG}"
echo "======================"

mkdir -p logs/vjepa21_nsys
mkdir -p "$(dirname "${NSYS_OUTPUT}")"

srun ${AUTO_RESUME} -l "${ARGS[@]}" \
    bash ${VJEPA21_DIR}/scripts/nsys_wrapper.sh \
    python ${VJEPA21_DIR}/scripts/run_train.py "${TRAINING_ARGS[@]}"
