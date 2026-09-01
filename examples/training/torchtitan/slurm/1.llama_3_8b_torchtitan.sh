#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --nodes=4 # number of nodes to use  # can also run with 1 node
#SBATCH --job-name=torchtitan #name of your job
#SBATCH --output=logs/%x_%j.out # logfile for stdout
#SBATCH --exclusive # job has exclusive use of the resource, no sharing

set -ex;

###########################
###### User Variables #####
###########################

GPUS_PER_NODE=8

###########################
## Environment Variables ##
###########################

export NCCL_DEBUG=INFO
export FI_PROVIDER=efa
export FI_EFA_USE_HUGE_PAGE=0
## Switching SYNC_MEMOPS to zero can boost throughput with FSDP
## Disables CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
## Reduces memory synchronizations
## https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export NCCL_SOCKET_IFNAME=^docker,lo,veth

## Set HuggingFace metadata timeout (in seconds) for large clusters
export HF_HUB_ETAG_TIMEOUT=60

###########################
####### Torch Dist  #######
###########################

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=$GPUS_PER_NODE
    --nnodes=$SLURM_JOB_NUM_NODES
    --rdzv_id=$SLURM_JOB_ID
    --rdzv_backend=c10d
    --rdzv_endpoint=$(hostname)
)

export TORCHRUN=./pt_torchtitan/bin/torchrun
export TRAIN_SCRIPT=./torchtitan/torchtitan/train.py

# Default config is the vendored copy in this repo. To run with FP8 +
# torch.compile, set CONFIG_FILE=$(dirname $0)/configs/llama3_8b_fp8_compile.toml
# before sbatch (or pass it via --export).
CONFIG_FILE=${CONFIG_FILE:-"$(dirname "$0")/configs/llama3_8b.toml"}

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi

srun  ${AUTO_RESUME} -l ${TORCHRUN} "${TORCHRUN_ARGS[@]}" $TRAIN_SCRIPT  --job.config_file ${CONFIG_FILE}
