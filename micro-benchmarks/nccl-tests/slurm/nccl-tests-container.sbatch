#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --job-name=nccl-all_reduce_perf # name of your job
#SBATCH --nodes=2 # number of nodes to use, 24 p4d(e) = 192 A100 GPUs
#SBATCH --ntasks-per-node 8 # Number of GPU per node
###SBATCH --gpus-per-node=8 # number of GPU we reserve. Uncomment for AWS ParallelCluster
#SBATCH --output %x_%j.out
#SBATCH --error %x_%j.err
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1

### Disable hyperthreading by setting the tasks per core to 1
#SBATCH --ntasks-per-core=1

###########################
###### User Variables #####
###########################


# default variables for Enroot
: "${APPS_PATH:=/fsx}"
: "${NCCL_TESTS_PATH:=/opt/nccl-tests/build}"
: "${IMAGE:=$APPS_PATH/nccl-tests.sqsh}"

## Set libfabric flags to use EFA
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1

## Set this flag for debugging EFA
#export FI_LOG_LEVEL=warn

## NCCL Environment variables
export NCCL_DEBUG=INFO


declare -a ARGS=(
    --container-image $IMAGE
)

#Get Hostname and Instance IDs
mpirun -N 1 bash -c 'echo $(hostname): $(cat /sys/devices/virtual/dmi/id/board_asset_tag | tr -d " ")'

# Run NCCL test
srun "${ARGS[@]}" --mpi=pmix --cpu-bind=none $NCCL_TESTS_PATH/all_reduce_perf -b 8 -e 16G -f 2 -g 1 -c 1 -n 100
