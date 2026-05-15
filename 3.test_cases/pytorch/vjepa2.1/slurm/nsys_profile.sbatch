#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# nsys profiling script for V-JEPA 2.1 on ParallelCluster p5en (H200 GPUs).
#
# Profiles rank 0 only using nsys_wrapper.sh.  All other ranks run
# un-instrumented to avoid large output files and profiling overhead.
#
# Environment variables (set before sbatch or use --export):
#   NSYS_PROFILE_DIR - Subfolder name under nsys/ for this run.
#                      Default: "baseline"
#
# Usage:
#   sbatch /fsx/${USER}/vjepa2.1/nsys_profile.sbatch

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --job-name=vjepa21-nsys
#SBATCH --output=logs/vjepa21_nsys/%A.out
#SBATCH --error=logs/vjepa21_nsys/%A.err
#SBATCH --exclusive
#SBATCH --partition=p5en
#SBATCH --time=01:00:00

set -euo pipefail

###########################
###### User Variables #####
###########################

VJEPA21_DIR=/fsx/${USER}/vjepa2.1

# Profiling output directory
NSYS_PROFILE_DIR="${NSYS_PROFILE_DIR:-baseline}"
export NSYS_OUTPUT=${VJEPA21_DIR}/nsys/${NSYS_PROFILE_DIR}/rank0

# nsys binary location on this container
export NSYS_BIN=/usr/local/bin/nsys

###########################
## Container Variables ####
###########################

export CONTAINER_IMAGE=${VJEPA21_DIR}/vjepa2.sqsh
export FSX_MOUNT=/fsx:/fsx
# V-JEPA 2.1 code overlay: mount code at /workspace
export CODE_MOUNT=${VJEPA21_DIR}/code:/workspace

###########################
## Environment Variables ##
###########################

export NCCL_DEBUG=INFO
export FI_PROVIDER=efa
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export LD_PRELOAD=/lib/x86_64-linux-gnu/libnccl.so.2
export NCCL_SOCKET_IFNAME=^docker,lo,veth

###########################
## Container Args #########
###########################

declare -a ARGS=(
    --container-image $CONTAINER_IMAGE
    --container-mounts "${FSX_MOUNT},${CODE_MOUNT}"
    --container-workdir /workspace
)

###########################
## Training Args ##########
###########################

declare -a TRAINING_ARGS=(
    --fname ${VJEPA21_DIR}/configs/benchmark-vitg-8nodes.yaml
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
echo "Config:      ${VJEPA21_DIR}/configs/benchmark-vitg-8nodes.yaml"
echo "======================"

mkdir -p logs/vjepa21_nsys
mkdir -p "$(dirname "${NSYS_OUTPUT}")"

/opt/slurm/bin/srun ${AUTO_RESUME} -l "${ARGS[@]}" \
    bash ${VJEPA21_DIR}/scripts/nsys_wrapper.sh \
    python ${VJEPA21_DIR}/scripts/run_train.py "${TRAINING_ARGS[@]}"
