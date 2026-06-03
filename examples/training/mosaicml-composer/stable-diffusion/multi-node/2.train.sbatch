#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --nodes=2 # number of nodes to use, 24 p4d(e) = 192 A100 GPUs
#SBATCH --job-name=mosaicml-stable-diffusion # name of your job
#SBATCH --gpus-per-node=8 # Number of GPU per node
#SBATCH --gres=gpu:8 # number of GPU we reserve
#SBATCH --gpus-per-task=8 # Number of GPU per node
#SBATCH --exclusive # job has exclusive use of the resource, no sharing
#SBATCH --wait-all-nodes=1
#SBATCH --output jobs/slurm-%j.out

# default variables for Enroot
: "${APPS_PATH:=/apps}"
: "${DATA_PATH:=/fsx}"

# default variables for Enroot
: "${IMAGE:=$APPS_PATH/mosaicml/stable-diffusion.sqsh}"
: "${FSX_MOUNT:=$DATA_PATH:$DATA_PATH}"

## Plenty of EFA level variables
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4d
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_PROVIDER=efa # change to eth if you want to use ENA for comparisons
export FI_EFA_ENABLE_SHM_TRANSFER=1
export NCCL_DEBUG=INFO
export WANDB_MODE=offline

declare -a ARGS=(
    --container-image $IMAGE
    --container-mounts $FSX_MOUNT
)

NODES=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
NNODES=${#NODES[@]}
NODES_ARRAY=($NODES)
HEAD_NODE=${NODES_ARRAY[0]}
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
MASTER_PORT=$RANDOM
NPROC=8
WORLD_SIZE=$((NNODES * NPROC))

srun -l "${ARGS[@]}" python -c "import streaming; streaming.base.util.clean_stale_shared_memory()"
function run_compose() {
    srun --nodelist=${NODE} --ntasks=1 -l "${ARGS[@]}" composer \
        --world_size ${WORLD_SIZE} \
        --nproc ${NPROC} \
        --node_rank ${NODE_RANK} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        --verbose \
        benchmark.py \
        --use_ema --use_synth_data --device_train_microbatch_size 4
}

NODE_RANK=1
for (( NODE_RANK=1; NODE_RANK<${NNODES}; NODE_RANK++ ))
do
    NODE=${NODES[$NODE_RANK]}
    echo "Run compute node ${NODE} for rank: ${NODE_RANK}"
    run_compose &
done

NODE_RANK=0
NODE=${HEAD_NODE}
echo "Run master node ${NODE} for rank: ${NODE_RANK}"
run_compose
wait
