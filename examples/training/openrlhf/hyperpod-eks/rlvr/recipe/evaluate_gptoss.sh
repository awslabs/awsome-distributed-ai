#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Post-training evaluation workflow for OpenRLHF GRPO checkpoint.
#
# Unlike veRL, OpenRLHF saves checkpoints in HuggingFace format directly,
# so NO conversion step is needed.
#
# Steps:
#   1. Run 50-question language compliance evaluation via vLLM
#   2. Optionally run SFT baseline comparison
#
# Usage:
#   # From a Ray worker pod (with GPUs):
#   bash evaluate_gptoss.sh
#
#   # Skip baseline comparison:
#   RUN_BASELINE=0 bash evaluate_gptoss.sh

set -euo pipefail

# ---- Configuration ----
GRPO_MODEL="${GRPO_MODEL:-/fsx/openrlhf/checkpoints/grpo-gpt-oss-20b}"
SFT_MODEL="${SFT_MODEL:-/fsx/models/gpt-oss-20b-sft-merged}"
OUTPUT_DIR="/fsx/experiments"
TP=4
RUN_BASELINE="${RUN_BASELINE:-1}"

echo "============================================="
echo "OpenRLHF GRPO Post-Training Evaluation"
echo "============================================="
echo "GRPO model:  ${GRPO_MODEL}"
echo "SFT model:   ${SFT_MODEL}"
echo "TP size:     ${TP}"
echo "============================================="

# ---- Step 1: Evaluate GRPO model ----
# OpenRLHF saves HF format directly — no conversion needed!
echo ""
echo "[Step 1] Evaluating GRPO model..."

if [ ! -d "${GRPO_MODEL}" ] || [ ! -f "${GRPO_MODEL}/config.json" ]; then
    echo "ERROR: GRPO model not found at ${GRPO_MODEL}"
    echo "Check that training completed and --save_hf_ckpt was used."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

python "$(dirname "$0")/evaluate_gptoss.py" \
    --model_path "${GRPO_MODEL}" \
    --tp "${TP}" \
    --gpu_mem 0.85 \
    --max_model_len 2048 \
    --max_tokens 512 \
    --output "${OUTPUT_DIR}/openrlhf_grpo_eval.txt"

echo "[Step 1] GRPO evaluation complete."

# ---- Step 2: Evaluate SFT baseline for comparison ----
if [ "${RUN_BASELINE}" = "1" ] && [ -d "${SFT_MODEL}" ]; then
    echo ""
    echo "[Step 2] Evaluating SFT baseline for comparison..."

    python "$(dirname "$0")/evaluate_gptoss.py" \
        --model_path "${SFT_MODEL}" \
        --tp "${TP}" \
        --gpu_mem 0.85 \
        --max_model_len 2048 \
        --max_tokens 512 \
        --output "${OUTPUT_DIR}/sft_baseline_eval.txt"

    echo "[Step 2] SFT baseline evaluation complete."
    echo ""
    echo "Compare results:"
    echo "  OpenRLHF GRPO: ${OUTPUT_DIR}/openrlhf_grpo_eval.txt"
    echo "  SFT baseline:  ${OUTPUT_DIR}/sft_baseline_eval.txt"
else
    echo "[Step 2] Skipping SFT baseline (RUN_BASELINE=0 or model not found)"
fi

echo ""
echo "============================================="
echo "Evaluation complete!"
echo "============================================="
