#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --nodes=2                   # number of nodes (e.g. 2 p6-b300.48xlarge = 16 B300 GPUs)
#SBATCH --job-name=megatron_llama3_8b
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1

set -exuo pipefail

##################################################
###### Llama 3 8B model architecture        ######
##################################################
# Reference: https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
declare -a MEGATRON_ARGS=(
    --num-layers 32
    --hidden-size 4096
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --ffn-hidden-size 14336

    # Native shape on 8xB300 per node (8 B300 GPUs, 275 GB HBM3e each).
    # Shard the optimizer state via DP rather than splitting the model.
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 2

    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

# Llama 3 architecture flags. Do not comment or remove.
MEGATRON_ARGS+=(
   --untie-embeddings-and-output-weights
   --position-embedding-type rope
   --rotary-base 500000
   --no-position-embedding
   --normalization RMSNorm
   --swiglu
   --no-masked-softmax-fusion
)

MEGATRON_ARGS+=(
    --use-flash-attn
    --transformer-impl transformer_engine
)


###########################
###### User Variables #####
###########################

: "${SEQ_LENGTH:=8192}"
: "${MAX_POSITION_EMBEDDINGS:=8192}"
: "${MICRO_BATCH_SIZE:=1}"
: "${GLOBAL_BATCH_SIZE:=512}"

# default variables for Enroot
: "${IMAGE:=$(pwd)/megatron-training.sqsh}"
: "${DATA_PATH:=/fsx}"
: "${FSX_MOUNT:=$(pwd):$DATA_PATH}"


###########################
## Environment Variables ##
###########################

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1


#########################
## Command and Options ##
#########################

declare -a ARGS=(
    --container-image $IMAGE
    --container-mounts $FSX_MOUNT
)

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=8
    --nnodes=$SLURM_JOB_NUM_NODES
    --rdzv_id=$SLURM_JOB_ID
    --rdzv_backend=c10d
    --rdzv_endpoint=$(hostname)
)

MEGATRON_ARGS+=(
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE

    --train-iters 200
    --split 100,0,0
)
[[ -f ${IMAGE} ]] || { echo "Could not find enroot image: $IMAGE" ; exit 1 ; }
srun -l "${ARGS[@]}" python -m torch.distributed.run "${TORCHRUN_ARGS[@]}" /workspace/Megatron-LM/pretrain_gpt.py \
        "${MEGATRON_ARGS[@]}" \
        --use-mcore-models \
        --log-throughput \
        --lr 3.0e-4 \
        --min-lr 3.0e-5 \
        --lr-decay-style cosine \
        --log-interval 1 \
        --eval-iters 0 \
        --eval-interval 1000 \
        --data-path ${DATA_PATH}/llama3/my-llama3_text_document \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model meta-llama/Meta-Llama-3-8B \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.006 \
        --bf16
