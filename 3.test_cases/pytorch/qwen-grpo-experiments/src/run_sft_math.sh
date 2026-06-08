#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Phase 2 - Arm B: SFT on math data (GSM8K + MATH)
# Qwen2.5-7B-Instruct + LoRA (r=16, alpha=32)
#
# Runs via torchrun on the Ray head node (single-node, 8 GPUs)
# Output: LoRA checkpoint at $CKPT_HOME/math/sft/

set -euo pipefail

# === Configuration ===
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
DATA_DIR="${DATA_DIR:-/fsx/qwen-grpo/data/math}"
CKPT_DIR="${CKPT_DIR:-/fsx/qwen-grpo/checkpoints/math/sft}"

TRAIN_FILE="${DATA_DIR}/sft_train.parquet"
VAL_FILE="${DATA_DIR}/sft_val.parquet"

SFT_LR="${SFT_LR:-2e-5}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
SFT_EPOCHS="${SFT_EPOCHS:-3}"
MICRO_BATCH="${SFT_MICRO_BATCH:-2}"

echo "============================================"
echo "Phase 2 - SFT (Math)"
echo "============================================"
echo "Model:      ${MODEL_PATH}"
echo "Data:       ${DATA_DIR}"
echo "Checkpoint: ${CKPT_DIR}"
echo "LR:         ${SFT_LR}"
echo "LoRA:       r=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "Epochs:     ${SFT_EPOCHS}"
echo "============================================"

# Pre-flight
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "ERROR: ${TRAIN_FILE} not found. Run data_preprocess_math.py first."
    exit 1
fi

mkdir -p "${CKPT_DIR}"

# Launch SFT
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu=${MICRO_BATCH} \
    data.max_length=2048 \
    \
    model.path="${MODEL_PATH}" \
    model.trust_remote_code=True \
    model.use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    model.lora_rank=${LORA_RANK} \
    model.lora_alpha=${LORA_ALPHA} \
    model.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
    \
    optim.lr=${SFT_LR} \
    optim.weight_decay=0.01 \
    optim.warmup_ratio=0.1 \
    optim.lr_scheduler_type=cosine \
    \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.total_epochs=${SFT_EPOCHS} \
    trainer.save_freq=200 \
    trainer.project_name="qwen-grpo-experiments" \
    trainer.experiment_name="phase2-sft-math" \
    trainer.logger='["console"]'

echo "============================================"
echo "SFT (Math) complete!"
echo "Checkpoint: ${CKPT_DIR}"
echo "============================================"
