#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH -N 2 # number of nodes to run the scrip on, use 2 here
#SBATCH --job-name=megatron_gpt # name of your job
#SBATCH --ntasks-per-node 1 # Number of tasks per node, we need one here
#SBATCH --gres=gpu:8 # number of GPU we reserve
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1

### Disable hyperthreading by setting the tasks per core to 1
#SBATCH --ntasks-per-core=1

set -ex

# Validate that mpirun does not need -x to propagate env vars defined in .sbatch script

###########################
###### User Variables #####
###########################

# default variables for Enroot
: "${APPS_PATH:=/apps}"
: "${IMAGE:=$APPS_PATH/pytorch-screen.sqsh}"
: "${FSX_MOUNT:=/fsx:/fsx}"
: "${SCREEN_PT_SCRIPT_PATH:=$PWD}"


declare -a ARGS=(
    --container-image $IMAGE
    --container-mount-home
    --container-mounts $FSX_MOUNT
)

echo "
Hostname: $(hostname)
"

env

/usr/bin/time srun -l "${ARGS[@]}" --mpi=pmix bash -c "
which nvidia-smi
nvidia-smi
which python
python --version
python ${SCREEN_PT_SCRIPT_PATH}/pytorch-screen.py
"
