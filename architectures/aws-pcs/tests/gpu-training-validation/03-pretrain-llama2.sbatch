#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Stage 3c: Tiny Llama-2 pretraining smoke test (2 nodes, 16 H100, 5 iters).
# Adapted from 3.test_cases/megatron/megatron-lm/slurm/llama2/pretrain-llama2.sbatch.
# Differences from upstream: llama2-7b preset (TP=1, PP=1) instead of 70b, and
# W&B logging removed.

#SBATCH --nodes=2
#SBATCH --job-name=megatron_llama2
#SBATCH --partition=gpu-p5
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1
#SBATCH --output=03-pretrain-llama2_%j.out
#SBATCH --error=03-pretrain-llama2_%j.out

set -euxo pipefail

##############################
###### Model: llama2-7b  #####
##############################
declare -a MEGATRON_ARGS=(
    --num-layers 32
    --hidden-size 4096
    --num-attention-heads 32

    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
)

# Required for Llama2-style architecture. Do not comment or remove.
MEGATRON_ARGS+=(
   --untie-embeddings-and-output-weights
   --position-embedding-type rope
   --no-position-embedding
   --normalization RMSNorm
   --swiglu
   --no-masked-softmax-fusion
)

# Flags to make it possible to test with as few nodes as possible
MEGATRON_ARGS+=(
    --no-rope-fusion
    --use-flash-attn
    --transformer-impl transformer_engine
)

###########################
###### User Variables #####
###########################
: "${SEQ_LENGTH:=4096}"
: "${MAX_POSITION_EMBEDDINGS:=4096}"
: "${MICRO_BATCH_SIZE:=1}"
: "${GLOBAL_BATCH_SIZE:=32}"

: "${IMAGE:=/fsx/validation/aws-megatron-lm.sqsh}"
: "${VALIDATION_DIR:=/fsx/validation}"
: "${DATA_PATH:=/fsx}"
: "${FSX_MOUNT:=${VALIDATION_DIR}:${DATA_PATH}}"

###########################
## Environment Variables ##
###########################
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NVLS_ENABLE=0
export NCCL_AVOID_RECORD_STREAMS=1          # torch<2.2
export TORCH_NCCL_AVOID_RECORD_STREAMS=1    # torch>=2.2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1

[[ -f "${IMAGE}" ]] || { echo "Missing ${IMAGE}; run 03-build-megatron.sbatch first." >&2; exit 1; }
[[ -f "${VALIDATION_DIR}/llama2/my-llama2_text_document.bin" ]] || {
  echo "Missing preprocessed data; run 03-data-preproc.sbatch first." >&2; exit 1; }

declare -a ARGS=(
    --container-image "${IMAGE}"
    --container-mounts "${FSX_MOUNT}"
)

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=8
    --nnodes="${SLURM_JOB_NUM_NODES}"
    --rdzv_id="${SLURM_JOB_ID}"
    --rdzv_backend=c10d
    --rdzv_endpoint="$(hostname)"
)

MEGATRON_ARGS+=(
    --seq-length "${SEQ_LENGTH}"
    --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --train-iters 5
    --split 100,0,0
)

srun -l "${ARGS[@]}" python -m torch.distributed.run "${TORCHRUN_ARGS[@]}" \
    /workspace/Megatron-LM/pretrain_gpt.py \
    "${MEGATRON_ARGS[@]}" \
    --use-mcore-models \
    --log-throughput \
    --lr 6.0e-5 \
    --min-lr 6.0e-6 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 0 \
    --eval-interval 1000 \
    --data-path "${DATA_PATH}/llama2/my-llama2_text_document" \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model "${DATA_PATH}/llama2/tokenizer.model" \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --fp16
