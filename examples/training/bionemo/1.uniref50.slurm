#!/bin/bash
#SBATCH --nodes=1                    # number of nodes
#SBATCH --ntasks-per-node=1          # n tasks per machine (one task per gpu) <required>
#SBATCH --cpus-per-task=16
#SBATCH --exclusive                   # exclusive node access
#SBATCH --output slurm-uniref-%j.out


###########################
###### User Variables #####
###########################

# default variables for Enroot
: "${IMAGE:=$(pwd)/${ENROOT_IMAGE}}"
: "${DATA_PATH:=/fsx}"
: "${FSX_MOUNT:=$DATA_PATH:$DATA_PATH}"

declare -a ARGS=(
    --container-image $IMAGE
    --container-mount-home
    --container-mounts $FSX_MOUNT
)

srun -l "${ARGS[@]}"  python3 /workspace/bionemo2/prepare_uniref50.py
