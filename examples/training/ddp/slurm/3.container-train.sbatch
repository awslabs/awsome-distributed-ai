#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --job-name=ddp-container
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1
#SBATCH --nodes 2
#SBATCH --output=logs/%x_%j.out # logfile for stdout/stderr

export LOGLEVEL=INFO
export NVIDIA_VISIBLE_DEVICES=void

declare -a ARGS=(
    --container-image ${PWD}/pytorch.sqsh
    --container-mounts $(dirname "$PWD")
)

# Use NPROC_PER_NODE environment variable if set, otherwise default to 4
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

declare -a TORCHRUN_ARGS=(
    --nproc_per_node="$NPROC_PER_NODE"     # For GPU: Set this to number of GPUs per node
    --nnodes=$SLURM_JOB_NUM_NODES
    --rdzv_id=$SLURM_JOB_ID
    --rdzv_backend=c10d
    --rdzv_endpoint=$(hostname)
)

declare -a TRAIN_ARGS=(
    --total_epochs 500
    --save_every 1
    --batch_size 32
    --checkpoint_path ./snapshot.pt
    --use_mlflow
)

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi

srun ${AUTO_RESUME} -l "${ARGS[@]}" torchrun \
    "${TORCHRUN_ARGS[@]}" \
    $(dirname "$PWD")/ddp.py ${TRAIN_ARGS[@]}
