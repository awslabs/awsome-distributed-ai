#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Post-training evaluation workflow for veRL GRPO checkpoint.
#
# Steps:
#   1. Convert FSDP shards → HuggingFace format (if not already done)
#   2. Run 50-question language compliance evaluation via vLLM
#   3. Optionally run SFT baseline comparison
#
# Usage:
#   # From the Ray head pod:
#   bash evaluate_gptoss.sh
#
#   # With custom step:
#   EVAL_STEP=60 bash evaluate_gptoss.sh
#
#   # Skip conversion (already done):
#   SKIP_CONVERT=1 bash evaluate_gptoss.sh

set -euo pipefail

# ---- Configuration ----
EVAL_STEP="${EVAL_STEP:-80}"
CKPT_BASE="/fsx/verl/ckpts/GRPO-GPT-OSS/GRPO-gpt-oss-20b-language"
CKPT_DIR="${CKPT_BASE}/global_step_${EVAL_STEP}/actor"
MERGED_DIR="/fsx/verl/merged_model/gpt-oss-20b-grpo-step${EVAL_STEP}"
SFT_MODEL="/fsx/verl/models/openai/gpt-oss-20b"
OUTPUT_DIR="/fsx/experiments"
TP=4
SKIP_CONVERT="${SKIP_CONVERT:-0}"
RUN_BASELINE="${RUN_BASELINE:-1}"

echo "============================================="
echo "veRL GRPO Post-Training Evaluation"
echo "============================================="
echo "Checkpoint step: ${EVAL_STEP}"
echo "Checkpoint dir:  ${CKPT_DIR}"
echo "Merged dir:      ${MERGED_DIR}"
echo "TP size:         ${TP}"
echo "============================================="

# ---- Step 1: Convert FSDP shards to HuggingFace format ----
if [ "${SKIP_CONVERT}" = "0" ]; then
    if [ -d "${MERGED_DIR}" ] && [ -f "${MERGED_DIR}/config.json" ]; then
        echo "[Step 1] Merged model already exists at ${MERGED_DIR}, skipping conversion."
    else
        echo "[Step 1] Converting FSDP checkpoint to HuggingFace format..."
        if [ ! -d "${CKPT_DIR}" ]; then
            echo "ERROR: Checkpoint directory not found: ${CKPT_DIR}"
            echo "Available checkpoints:"
            ls -d ${CKPT_BASE}/global_step_* 2>/dev/null || echo "  (none)"
            exit 1
        fi

        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "${CKPT_DIR}" \
            --target_dir "${MERGED_DIR}"

        echo "[Step 1] Conversion complete. Model saved to: ${MERGED_DIR}"
    fi
else
    echo "[Step 1] Skipping conversion (SKIP_CONVERT=1)"
fi

# ---- Step 2: Evaluate GRPO model ----
echo ""
echo "[Step 2] Evaluating GRPO model (step ${EVAL_STEP})..."
mkdir -p "${OUTPUT_DIR}"

python "$(dirname "$0")/evaluate_gptoss.py" \
    --model_path "${MERGED_DIR}" \
    --tp "${TP}" \
    --gpu_mem 0.85 \
    --max_model_len 2048 \
    --max_tokens 512 \
    --output "${OUTPUT_DIR}/grpo_eval_step${EVAL_STEP}.txt"

echo "[Step 2] GRPO evaluation complete."

# ---- Step 3: Evaluate SFT baseline for comparison ----
if [ "${RUN_BASELINE}" = "1" ] && [ -d "${SFT_MODEL}" ]; then
    echo ""
    echo "[Step 3] Evaluating SFT baseline for comparison..."

    python "$(dirname "$0")/evaluate_gptoss.py" \
        --model_path "${SFT_MODEL}" \
        --tp "${TP}" \
        --gpu_mem 0.85 \
        --max_model_len 2048 \
        --max_tokens 512 \
        --output "${OUTPUT_DIR}/sft_baseline_eval.txt"

    echo "[Step 3] SFT baseline evaluation complete."
    echo ""
    echo "Compare results:"
    echo "  GRPO: ${OUTPUT_DIR}/grpo_eval_step${EVAL_STEP}.txt"
    echo "  SFT:  ${OUTPUT_DIR}/sft_baseline_eval.txt"
else
    echo "[Step 3] Skipping SFT baseline (RUN_BASELINE=0 or model not found)"
fi

echo ""
echo "============================================="
echo "Evaluation complete!"
echo "============================================="
