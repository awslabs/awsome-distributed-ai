#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -euo pipefail
echo "=============================================="
echo "NeMo RL GRPO Training Entrypoint"
echo "=============================================="

# --- Configurable paths (override via environment variables) ---
SHARED_DIR="${SHARED_DIR:-/fsx}"
GRPO_MODEL="${GRPO_MODEL:-nvidia/Nemotron-Mini-4B-Instruct}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-50}"
GRPO_NUM_NODES="${GRPO_NUM_NODES:-2}"
CKPT_DIR="${CKPT_DIR:-${SHARED_DIR}/phase2-checkpoints}"
DATA_PATH="${DATA_PATH:-${SHARED_DIR}/goldilocks/train.jsonl}"

# --- Ray IP configuration ---
# Ray handles IP discovery via its GCS; no /etc/hosts manipulation needed.
# The Ray head address is set automatically by the RayJob controller.

# --- NCCL environment ---
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-ofi}"
export NCCL_TUNER_PLUGIN="${NCCL_TUNER_PLUGIN:-ofi}"
export LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}"

GPU_COUNT=$(nvidia-smi -L | wc -l)

# Only clear checkpoints on first run -- preserve for fault recovery resume
if [ "${CLEAR_CHECKPOINTS:-0}" = "1" ]; then
  echo "Clearing checkpoints (CLEAR_CHECKPOINTS=1)"
  rm -rf "${CKPT_DIR:?}"/*
fi
mkdir -p "$CKPT_DIR"
EXISTING=$(ls -d "$CKPT_DIR"/step_* 2>/dev/null | wc -l)
[ "$EXISTING" -gt 0 ] && echo "Found $EXISTING existing checkpoint(s) -- will resume from latest"

export HF_HOME="${HF_HOME:-${SHARED_DIR}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SHARED_DIR}/hf_datasets}"
export TORCH_HOME="${TORCH_HOME:-${SHARED_DIR}/torch_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SHARED_DIR}/triton_cache}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR"

# --- Determine the base model to train from ---
# If a pre-merged model exists on shared storage, use it directly.
# Otherwise, fall back to the HuggingFace model name.
MERGED="${MERGED_MODEL_PATH:-${SHARED_DIR}/nemotron-phase1-merged}"
if [ -f "$MERGED/model.safetensors" ]; then
  echo "Using pre-merged model at $MERGED"
  TRAIN_MODEL="$MERGED"
else
  echo "No pre-merged model found at $MERGED; using HuggingFace model: $GRPO_MODEL"
  TRAIN_MODEL="$GRPO_MODEL"
fi

cd /opt/nemo-rl

echo "  Model: $TRAIN_MODEL"
echo "  Nodes: $GRPO_NUM_NODES, GPUs/node: $GPU_COUNT"
echo "  Max steps: $GRPO_MAX_STEPS"
echo "  Data: $DATA_PATH"

exec python3 examples/run_grpo.py \
    cluster.num_nodes="$GRPO_NUM_NODES" \
    cluster.gpus_per_node="$GPU_COUNT" \
    grpo.max_num_steps="$GRPO_MAX_STEPS" \
    grpo.max_num_epochs=2 \
    grpo.num_prompts_per_step=16 \
    grpo.num_generations_per_prompt=16 \
    grpo.use_dynamic_sampling=true \
    grpo.batch_multiplier=4 \
    grpo.dynamic_sampling_max_gen_batches=10 \
    grpo.overlong_filtering=true \
    policy.model_name="$TRAIN_MODEL" \
    policy.tokenizer.name="$GRPO_MODEL" \
    policy.megatron_cfg.enabled=false \
    policy.dtensor_cfg.enabled=true \
    policy.dtensor_cfg._v2=true \
    policy.dtensor_cfg.tensor_parallel_size=1 \
    policy.dtensor_cfg.activation_checkpointing=true \
    policy.dtensor_cfg.cpu_offload=true \
    policy.dtensor_cfg.lora_cfg.enabled=true \
    policy.dtensor_cfg.lora_cfg.dim=32 \
    policy.dtensor_cfg.lora_cfg.alpha=64 \
    policy.dtensor_cfg.lora_cfg.dropout=0.0 \
    policy.dtensor_cfg.lora_cfg.dropout_position=pre \
    policy.dtensor_cfg.lora_cfg.lora_A_init=xavier \
    'policy.dtensor_cfg.lora_cfg.target_modules=[q_proj,k_proj,v_proj,o_proj]' \
    'policy.dtensor_cfg.lora_cfg.exclude_modules=[]' \
    policy.optimizer.kwargs.lr=5e-5 \
    policy.train_global_batch_size=256 \
    policy.train_micro_batch_size=16 \
    policy.max_total_sequence_length=1024 \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.4 \
    ++loss_fn.reference_policy_kl_penalty=0.04 \
    ++data.train.dataset_name=ResponseDataset \
    ++data.train.data_path="$DATA_PATH" \
    ++data.train.input_key=problem \
    ++data.train.output_key=answer \
    ++data.train.split_validation_size=0.05 \
    logger.wandb_enabled=false \
    logger.log_dir=/tmp/grpo-logs \
    checkpointing.enabled=true \
    checkpointing.checkpoint_dir="$CKPT_DIR" \
    checkpointing.save_period=25 \
    checkpointing.keep_top_k=5
