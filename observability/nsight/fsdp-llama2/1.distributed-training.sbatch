#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --nodes=2 # number of nodes to use
#SBATCH --job-name=FSDP # name of your job
#SBATCH --exclusive # job has exclusive use of the resource, no sharing

set -ex;

###########################
###### User Variables #####
###########################

GPUS_PER_NODE=8 # 4 for G5.12x, 8 for P4/P5

###########################
## Environment Variables ##
###########################

## Plenty of EFA level variables
## Comment out for non-efa instances (G4d, P3)
## For G5.12x, Comment out RDMA and Fork safe
## For G4dn and other G5, comment out all
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4d
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_PROVIDER=efa
#export NCCL_DEBUG=INFO

###########################
####### Torch Dist  #######
###########################

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(hostname) \
)

source /home/ubuntu/.bashrc
source activate fsdp

export TRAIN_SCRIPT=./train.py

############################
# Llama 2 Training Params ##
############################

declare -a TRAINING_ARGS=(
    --max_context_width=4096 \
    --num_key_value_heads=32 \ # 7b: 32 13b: 40 70b: 8
    --llama_intermediate_size=11008 \ # 7b: 11008 13b: 13824 70b: 28672
    --hidden_width=4096 \ # 7b: 4096 13b: 5120 70b: 8192
    --num_layers=32 \ # 7b: 32 13b: 40 70b: 80
    --num_heads=32 \ # 7b: 32 13b: 40 70b: 64
    --model_type=llama_v2 \
    --tokenizer="hf-internal-testing/llama-tokenizer" \
    --checkpoint_freq=5000 \
    --validation_freq=500 \
    --max_steps 80 \
    --checkpoint_dir=./checkpoints \
    --dataset='c4' \
    --dataset_config_name='en' \
    --resume_from_checkpoint=./checkpoints \
    --train_batch_size=1 \
    --val_batch_size=1 \
    --sharding_strategy="hybrid" \
    --epochs=1
    --nsys_start_step=70
    --nsys_end_step=75
)

srun -l /fsx/nsys-slurm-exec torchrun "${TORCHRUN_ARGS[@]}" $TRAIN_SCRIPT "${TRAINING_ARGS[@]}"