#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# ============================================================
# SLIME GRPO Training — Qwen3-4B on HyperPod EKS (Colocated)
#
# This script submits a Ray job to run GRPO training using SLIME
# with Megatron-LM training and SGLang inference on shared GPUs.
#
# Prerequisites:
#   - Ray cluster deployed via kubernetes/raycluster.yaml
#   - Model downloaded and converted to torch_dist format
#   - Training data on FSx
#   - source env_vars before running
#
# Usage:
#   source env_vars
#   bash recipe/run_grpo_qwen3_4b.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

# Source environment if not already loaded. Override the file with ENV_FILE,
# e.g.  ENV_FILE=env_vars.disaggregated bash recipe/run_grpo_qwen3_4b.sh
if [[ -z "${MODEL_LOCAL:-}" ]]; then
    ENV_FILE="${ENV_FILE:-${PROJECT_DIR}/env_vars}"
    echo "[INFO] Sourcing ${ENV_FILE}..."
    source "${ENV_FILE}"
fi

# Validate required variables
for var in MODEL_LOCAL MODEL_DIST PROMPT_DATA CHECKPOINT_DIR; do
    if [[ -z "${!var:-}" ]]; then
        echo "[ERROR] ${var} is not set. Please configure env_vars."
        exit 1
    fi
done

echo "============================================================"
echo "  SLIME GRPO Training — Qwen3-4B (Colocated Mode)"
echo "============================================================"
echo "  Model:          ${MODEL_LOCAL}"
echo "  Megatron ckpt:  ${MODEL_DIST}"
echo "  Training data:  ${PROMPT_DATA}"
echo "  Checkpoints:    ${CHECKPOINT_DIR}/qwen3-4b-grpo/"
echo "  Nodes:          ${ACTOR_NUM_NODES} x ${ACTOR_GPUS_PER_NODE} GPUs"
echo "  Colocated:      ${COLOCATE}"
echo "  Rollout BS:     ${ROLLOUT_BATCH_SIZE} x ${N_SAMPLES_PER_PROMPT}"
echo "  Global BS:      ${GLOBAL_BATCH_SIZE}"
echo "  Num rollouts:   ${NUM_ROLLOUT}"
echo "============================================================"

# Build the training command
# When RM_TYPE=remote_rm, point SLIME at the CPU-hosted reward Service via
# --rm-url (see kubernetes/reward-service.yaml). Otherwise scoring is in-process.
RM_ARGS="--rm-type ${RM_TYPE}"
if [ "${RM_TYPE}" = "remote_rm" ]; then
    if [ -z "${RM_URL}" ]; then
        echo "[ERROR] RM_TYPE=remote_rm but RM_URL is not set. Configure it in env_vars."
        exit 1
    fi
    RM_ARGS="${RM_ARGS} --rm-url ${RM_URL}"
    echo "  Reward:         remote_rm @ ${RM_URL}"
fi

TRAIN_CMD="cd /opt/slime && source scripts/models/${MODEL_SCRIPT} && python3 train.py \
    \${MODEL_ARGS[@]} \
    --hf-checkpoint ${MODEL_LOCAL} \
    --ref-load ${MODEL_DIST} \
    --load ${CHECKPOINT_DIR}/qwen3-4b-grpo/ \
    --save ${CHECKPOINT_DIR}/qwen3-4b-grpo/ \
    --save-interval ${SAVE_INTERVAL} \
    \
    --prompt-data ${PROMPT_DATA} \
    --input-key prompt \
    --label-key label \
    --apply-chat-template \
    --rollout-shuffle \
    \
    ${RM_ARGS} \
    \
    --num-rollout ${NUM_ROLLOUT} \
    --rollout-batch-size ${ROLLOUT_BATCH_SIZE} \
    --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT} \
    --num-steps-per-rollout ${NUM_STEPS_PER_ROLLOUT:-1} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    \
    --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN} \
    --rollout-temperature ${ROLLOUT_TEMPERATURE} \
    --balance-data \
    \
    --eval-interval 10 \
    --eval-prompt-data aime ${EVAL_DATA} \
    --n-samples-per-eval-prompt 8 \
    --eval-max-response-len 16384 \
    --eval-top-p 1 \
    \
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    --context-parallel-size ${CP_SIZE} \
    --expert-model-parallel-size ${EP_SIZE} \
    --sequence-parallel \
    \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU} \
    \
    --advantage-estimator grpo \
    --use-kl-loss \
    --kl-loss-coef 0.00 \
    --kl-loss-type low_var_kl \
    --entropy-coef 0.00 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    \
    --optimizer adam \
    --lr ${LEARNING_RATE} \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    \
    --actor-num-nodes ${ACTOR_NUM_NODES} \
    --actor-num-gpus-per-node ${ACTOR_GPUS_PER_NODE} \
    --colocate \
    --rollout-num-gpus-per-engine ${ROLLOUT_GPUS_PER_ENGINE} \
    \
    --sglang-mem-fraction-static 0.8 \
    --sglang-log-level WARN"

# Submit via Ray job API
echo "[INFO] Submitting Ray job..."

ray job submit \
    --address="http://127.0.0.1:8265" \
    --runtime-env-json="{
        \"env_vars\": {
            \"PYTHONPATH\": \"/opt/Megatron-LM\",
            \"HF_TOKEN\": \"${HF_TOKEN}\",
            \"TOKENIZERS_PARALLELISM\": \"false\",
            \"NCCL_DEBUG\": \"WARN\",
            \"FI_PROVIDER\": \"efa\",
            \"FI_EFA_USE_DEVICE_RDMA\": \"1\"
        }
    }" \
    -- bash -c "${TRAIN_CMD}"

echo "[INFO] Job submitted. Monitor at http://localhost:8265"
