#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --nodes=2 # number of nodes to use, 2 p4d(e) = 16 A100 GPUs
#SBATCH --job-name=megatron_gpt # name of your job
#SBATCH --exclusive # job has exclusive use of the resource, no sharing
#SBATCH --wait-all-nodes=1

set -ex;

###########################
###### User Variables #####
###########################

GPUS_PER_NODE=${GPUS_PER_NODE:-8} # 4 for g5.12xlarge, 8 for p4de/p5/p5en
NODES=${SLURM_JOB_NUM_NODES:-2}

# configure TP/PP/GBS based on node count
if [ "$NODES" -le 4 ]; then
    : "${TENSOR_PARALLEL:=4}"
    : "${PIPELINE_PARALLEL:=2}"
    : "${GLOBAL_BATCH_SIZE:=288}"
elif [ $NODES -ge 8 ]; then
    : "${TENSOR_PARALLEL:=8}"
    : "${PIPELINE_PARALLEL:=4}"
    TOTAL_GPUS=$((NODES * GPUS_PER_NODE))
    DATA_PARALLEL=$((TOTAL_GPUS / (TENSOR_PARALLEL * PIPELINE_PARALLEL)))
    : "${GLOBAL_BATCH_SIZE:=$((DATA_PARALLEL * 576))}"
fi

# Model parameters, defaults to 39B model
# Refer to page 8 of this paper on how to tune models parameters
# https://arxiv.org/pdf/2104.04473.pdf
: "${NUM_LAYERS:=36}"
: "${HIDDEN_SIZE:=4096}"
: "${NUM_ATTENTION_HEADS:=32}"
: "${SEQ_LENGTH:=2048}"
: "${MAX_POSITION_EMBEDDINGS:=2048}"
: "${MICRO_BATCH_SIZE:=1}"


# default variables for Enroot
: "${IMAGE:=$(pwd)/megatron-training.sqsh}"
: "${DATA_PATH:=/fsx}"
: "${FSX_MOUNT:=/fsx:/fsx}"

###########################
## Environment Variables ##
###########################

# https://discuss.pytorch.org/t/nccl-network-is-unreachable-connection-refused-when-initializing-ddp/137352
# https://github.com/pytorch/pytorch/issues/68893
#export NCCL_SOCKET_IFNAME=ens
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

# async runtime error ...
export CUDA_DEVICE_MAX_CONNECTIONS=1

#########################
## Command and Options ##
#########################

declare -a ARGS=(
    --container-image $IMAGE
    --container-mounts $FSX_MOUNT
)

declare -a TORCHRUN_ARGS=(
    # change this to match the number of gpus per node:
    --nproc_per_node=$GPUS_PER_NODE
    --nnodes=$SLURM_JOB_NUM_NODES
    --rdzv_id=$SLURM_JOB_ID
    --rdzv_backend=c10d
    --rdzv_endpoint=$(hostname)
)

declare -a MEGATRON_ARGS=(
        --num-layers $NUM_LAYERS
        --hidden-size $HIDDEN_SIZE
        --num-attention-heads $NUM_ATTENTION_HEADS
        --seq-length $SEQ_LENGTH
        --max-position-embeddings $MAX_POSITION_EMBEDDINGS
        --micro-batch-size $MICRO_BATCH_SIZE
        --global-batch-size $GLOBAL_BATCH_SIZE
)

declare -a MEGATRON_PARALLELISM=(
        --tensor-model-parallel-size $TENSOR_PARALLEL
        --pipeline-model-parallel-size $PIPELINE_PARALLEL
)

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi

srun ${AUTO_RESUME} -l "${ARGS[@]}" python -m torch.distributed.run "${TORCHRUN_ARGS[@]}" /workspace/Megatron-LM/pretrain_gpt.py \
        "${MEGATRON_PARALLELISM[@]}" \
        "${MEGATRON_ARGS[@]}" \
        --train-samples 146484375 \
        --lr-decay-samples 126953125 \
        --lr-warmup-samples 183105 \
        --lr 6.0e-5 \
        --min-lr 6.0e-6 \
        --lr-decay-style cosine \
        --log-interval 1 \
        --eval-iters 40 \
        --eval-interval 1000 \
        --data-path "${DATA_PATH}/gpt2/my-gpt2_text_document" \
        --vocab-file "${DATA_PATH}/gpt2/gpt2-vocab.json" \
        --merge-file "${DATA_PATH}/gpt2/gpt2-merges.txt" \
        --split 98,2,0 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.006 \
        --fp16 \
        --recompute-activations \
