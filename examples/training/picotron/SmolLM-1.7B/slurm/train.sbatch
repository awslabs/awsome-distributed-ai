#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --nodes=2 # number of nodes to use
#SBATCH --job-name=picotron # name of your job
#SBATCH --output=logs/%x_%j.out # logfile for stdout
#SBATCH --exclusive # job has exclusive use of the resource, no sharing

set -ex;

###########################
###### User Variables #####
###########################

GPUS_PER_NODE=1

###########################
######## EFA / NCCL #######
###########################

## See architectures/efa-cheatsheet.md for full reference
export FI_EFA_USE_HUGE_PAGE=0
export FI_PROVIDER=efa
export NCCL_SOCKET_IFNAME=^docker,lo,veth

###########################
##### Container Args ######
###########################

declare -a SRUN_ARGS=(
    --container-image  $PWD/picotron.sqsh
    --container-mounts "$PWD"
    --container-workdir ${PWD}
)

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

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi

srun ${AUTO_RESUME} -l "${SRUN_ARGS[@]}" torchrun "${TORCHRUN_ARGS[@]}" \
    /picotron/train.py --config ${PWD}/conf/llama-1B-tp2/config.json