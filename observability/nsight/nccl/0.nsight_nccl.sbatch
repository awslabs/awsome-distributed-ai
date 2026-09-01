#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH -N 2 # number of nodes to use, 24 p4d(e) = 192 A100 GPUs
#SBATCH --job-name=megatron_gpt # name of your job
#SBATCH --ntasks-per-node 8 # Number of GPU per node
#SBATCH --gres=gpu:8 # number of GPU we reserve
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1

### Disable hyperthreading by setting the tasks per core to 1
#SBATCH --ntasks-per-core=1

###########################
###### User Variables #####
###########################

# default variables for Enroot
: "${APPS_PATH:=/apps}"
: "${NCCL_TESTS_PATH:=/opt/nccl-tests/build}"
: "${DATA_PATH:=/fsx}"
: "${FSX_MOUNT:=$DATA_PATH:$DATA_PATH}"

: "${IMAGE:=$APPS_PATH/nccl.sqsh}"

## Plenty of EFA level variables
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4d
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_PROVIDER=efa # change to eth if you want to use ENA for comparisons
export FI_EFA_ENABLE_SHM_TRANSFER=1
# https://discuss.pytorch.org/t/nccl-network-is-unreachable-connection-refused-when-initializing-ddp/137352
# https://github.com/pytorch/pytorch/issues/68893
#export NCCL_SOCKET_IFNAME=ens
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

declare -a ARGS=(
    --container-image $IMAGE
    --container-mount-home
    --container-mounts $FSX_MOUNT
    --no-container-remap-root
)

srun -l "${ARGS[@]}" --mpi=pmix /fsx/nccl-slurm-exec $NCCL_TESTS_PATH/scatter_perf -b 8 -e 2G -f 2 -g 1 -c 1 -n 100