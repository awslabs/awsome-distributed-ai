#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Phase 1 - Arm C/D: GRPO on multilingual data using veRL
# Qwen2.5-7B-Instruct + LoRA (r=16, alpha=32)
#
# Arm C: GRPO from base (MODEL_PATH = base model)
# Arm D: GRPO from SFT checkpoint (MODEL_PATH = merged SFT model)
#
# Submits as a Ray job for multi-node GRPO training.
# Output: LoRA checkpoint at $CKPT_DIR

set -euo pipefail

# === Configuration ===
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
DATA_DIR="${DATA_DIR:-/fsx/qwen-grpo/data/multilingual}"
CKPT_DIR="${CKPT_DIR:-/fsx/qwen-grpo/checkpoints/multilingual/grpo}"
REWARD_PATH="${REWARD_PATH:-/workspace/experiments/src/language_reward.py}"

TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_FILE="${DATA_DIR}/val.parquet"

GRPO_LR="${GRPO_LR:-1e-6}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
GRPO_EPOCHS="${GRPO_EPOCHS:-3}"
N_SAMPLES="${GRPO_N_SAMPLES:-8}"
TEMPERATURE="${GRPO_TEMPERATURE:-0.7}"
TRAIN_BATCH="${GRPO_TRAIN_BATCH:-64}"

# Cluster
NNODES="${NUM_NODES:-2}"
GPUS_PER_NODE="${NUM_GPU_PER_NODE:-8}"
RAY_ADDRESS="${RAY_ADDRESS:-http://localhost:8265}"

# Experiment naming
EXP_NAME="${EXP_NAME:-phase1-grpo-multilingual}"

echo "============================================"
echo "Phase 1 - GRPO (Multilingual)"
echo "============================================"
echo "Model:       ${MODEL_PATH}"
echo "Data:        ${DATA_DIR}"
echo "Checkpoint:  ${CKPT_DIR}"
echo "Reward:      ${REWARD_PATH}"
echo "LR:          ${GRPO_LR}"
echo "LoRA:        r=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "Epochs:      ${GRPO_EPOCHS}"
echo "N samples:   ${N_SAMPLES}"
echo "Temperature: ${TEMPERATURE}"
echo "Nodes:       ${NNODES} x ${GPUS_PER_NODE} GPUs"
echo "============================================"

# Pre-flight
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "ERROR: ${TRAIN_FILE} not found. Run data_preprocess_multilingual.py first."
    exit 1
fi

mkdir -p "${CKPT_DIR}"

# Submit GRPO via Ray
ray job submit --address="${RAY_ADDRESS}" --no-wait \
    --working-dir /workspace/experiments \
    -- python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.train_batch_size=${TRAIN_BATCH} \
    data.filter_overlong_prompts=True \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=${LORA_RANK} \
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
    actor_rollout_ref.model.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
    \
    actor_rollout_ref.actor.optim.lr=${GRPO_LR} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${N_SAMPLES} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    \
    custom_reward_function.path="${REWARD_PATH}" \
    custom_reward_function.name=compute_score \
    \
    algorithm.use_kl_in_reward=False \
    \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.total_epochs=${GRPO_EPOCHS} \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.project_name="qwen-grpo-experiments" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.logger='["console"]'

echo "============================================"
echo "GRPO job submitted to Ray: ${EXP_NAME}"
echo "Monitor: ${RAY_ADDRESS}"
echo "============================================"
