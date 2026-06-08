#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Master orchestration script for Qwen2.5-7B GRPO experiments
#
# Runs the full 2-phase experiment pipeline:
#   Phase 1: Multilingual (verify GRPO works on Qwen where it failed on GPT-OSS)
#   Phase 2: Math reasoning (demonstrate SFT+GRPO > SFT-only)
#
# Usage:
#   ./run_all.sh                    # Run everything
#   ./run_all.sh --phase 1          # Phase 1 only
#   ./run_all.sh --phase 2          # Phase 2 only
#   ./run_all.sh --step preprocess  # Only data preprocessing
#   ./run_all.sh --step sft         # Only SFT training
#   ./run_all.sh --step grpo        # Only GRPO training
#   ./run_all.sh --step eval        # Only evaluation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"

# Load environment
if [ -f "${SCRIPT_DIR}/k8s/env_vars" ]; then
    source "${SCRIPT_DIR}/k8s/env_vars"
fi

# Parse arguments
PHASE="${1:-all}"
STEP="${2:-all}"

case "${PHASE}" in
    --phase) PHASE="${2:-all}"; STEP="${3:-all}" ;;
    --step)  STEP="${2:-all}"; PHASE="all" ;;
esac

echo "========================================================"
echo "  Qwen2.5-7B GRPO Experiment Pipeline"
echo "========================================================"
echo "  Phase: ${PHASE}"
echo "  Step:  ${STEP}"
echo "  Time:  $(date)"
echo "========================================================"

# === Shared config ===
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
export DATA_HOME="${DATA_HOME:-/fsx/qwen-grpo}"
export CKPT_HOME="${CKPT_HOME:-/fsx/qwen-grpo/checkpoints}"
export RESULTS_DIR="${RESULTS_DIR:-/fsx/qwen-grpo/results}"

mkdir -p "${DATA_HOME}" "${CKPT_HOME}" "${RESULTS_DIR}"

# ========================================================================
# PHASE 1: MULTILINGUAL (Verification)
# ========================================================================

run_phase1() {
    echo ""
    echo "========================================================"
    echo "  PHASE 1: Multilingual Language Compliance"
    echo "========================================================"

    local DATA_DIR="${DATA_HOME}/data/multilingual"

    # --- Step 1: Data Preprocessing ---
    if [[ "${STEP}" == "all" || "${STEP}" == "preprocess" ]]; then
        echo ""
        echo "--- Phase 1: Data Preprocessing ---"
        python3 "${SRC_DIR}/data_preprocess_multilingual.py" \
            --model_name "${MODEL_PATH}" \
            --output_dir "${DATA_DIR}" \
            --eval_size 50
    fi

    # --- Step 2: Evaluate Base Model (Arm A) ---
    if [[ "${STEP}" == "all" || "${STEP}" == "eval" ]]; then
        echo ""
        echo "--- Phase 1 Arm A: Base Model Evaluation ---"
        python3 "${SRC_DIR}/evaluate_multilingual.py" \
            --model_path "${MODEL_PATH}" \
            --output_file "${RESULTS_DIR}/phase1_A_base.json" \
            --model_name "Arm A: Base Qwen2.5-7B-Instruct"
    fi

    # --- Step 3: SFT Training (Arm B) ---
    if [[ "${STEP}" == "all" || "${STEP}" == "sft" ]]; then
        echo ""
        echo "--- Phase 1 Arm B: SFT Training ---"
        export DATA_DIR="${DATA_DIR}"
        export CKPT_DIR="${CKPT_HOME}/multilingual/sft"
        bash "${SRC_DIR}/run_sft_multilingual.sh"
    fi

    # --- Step 4: Evaluate SFT (Arm B) ---
    if [[ "${STEP}" == "all" || "${STEP}" == "eval" ]]; then
        echo ""
        echo "--- Phase 1 Arm B: SFT Evaluation ---"
        python3 "${SRC_DIR}/evaluate_multilingual.py" \
            --model_path "${MODEL_PATH}" \
            --adapter_path "${CKPT_HOME}/multilingual/sft" \
            --output_file "${RESULTS_DIR}/phase1_B_sft.json" \
            --model_name "Arm B: SFT only"
    fi

    # --- Step 5: GRPO from Base (Arm C) ---
    if [[ "${STEP}" == "all" || "${STEP}" == "grpo" ]]; then
        echo ""
        echo "--- Phase 1 Arm C: GRPO from Base ---"
        export DATA_DIR="${DATA_DIR}"
        export CKPT_DIR="${CKPT_HOME}/multilingual/grpo-from-base"
        export EXP_NAME="phase1-grpo-from-base"
        bash "${SRC_DIR}/run_grpo_multilingual.sh"
    fi

    # --- Step 6: Merge SFT checkpoint for Arm D ---
    if [[ "${STEP}" == "all" || "${STEP}" == "grpo" ]]; then
        echo ""
        echo "--- Phase 1: Merging SFT checkpoint for Arm D ---"
        python3 "${SRC_DIR}/merge_lora_checkpoint.py" \
            --base_model "${MODEL_PATH}" \
            --adapter_path "${CKPT_HOME}/multilingual/sft" \
            --output_path "${DATA_HOME}/models/qwen-sft-multilingual-merged"
    fi

    # --- Step 7: GRPO from SFT (Arm D) ---
    if [[ "${STEP}" == "all" || "${STEP}" == "grpo" ]]; then
        echo ""
        echo "--- Phase 1 Arm D: GRPO from SFT ---"
        export MODEL_PATH="${DATA_HOME}/models/qwen-sft-multilingual-merged"
        export DATA_DIR="${DATA_DIR}"
        export CKPT_DIR="${CKPT_HOME}/multilingual/grpo-from-sft"
        export EXP_NAME="phase1-grpo-from-sft"
        bash "${SRC_DIR}/run_grpo_multilingual.sh"
        export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"  # Reset
    fi

    # --- Step 8: Evaluate All Arms ---
    if [[ "${STEP}" == "all" || "${STEP}" == "eval" ]]; then
        echo ""
        echo "--- Phase 1: Batch Evaluation ---"
        python3 "${SRC_DIR}/batch_eval.py" --phase multilingual --results_dir "${RESULTS_DIR}"
    fi
}

# ========================================================================
# PHASE 2: MATH REASONING (Demonstration)
# ========================================================================

run_phase2() {
    echo ""
    echo "========================================================"
    echo "  PHASE 2: Math Reasoning (GSM8K)"
    echo "========================================================"

    local DATA_DIR="${DATA_HOME}/data/math"

    # --- Step 1: Data Preprocessing ---
    if [[ "${STEP}" == "all" || "${STEP}" == "preprocess" ]]; then
        echo ""
        echo "--- Phase 2: Data Preprocessing ---"
        python3 "${SRC_DIR}/data_preprocess_math.py" \
            --model_name "${MODEL_PATH}" \
            --output_dir "${DATA_DIR}" \
            --datasets gsm8k
    fi

    # --- Step 2: Evaluate Base Model (Arm A) ---
    if [[ "${STEP}" == "all" || "${STEP}" == "eval" ]]; then
        echo ""
        echo "--- Phase 2 Arm A: Base Model Evaluation ---"
        python3 "${SRC_DIR}/evaluate_math.py" \
            --model_path "${MODEL_PATH}" \
            --output_file "${RESULTS_DIR}/phase2_A_base.json" \
            --model_name "Arm A: Base Qwen2.5-7B-Instruct" \
            --dataset gsm8k
    fi

    # --- Step 3: SFT Training (Arm B) ---
    if [[ "${STEP}" == "all" || "${STEP}" == "sft" ]]; then
        echo ""
        echo "--- Phase 2 Arm B: SFT Training ---"
        export DATA_DIR="${DATA_DIR}"
        export CKPT_DIR="${CKPT_HOME}/math/sft"
        bash "${SRC_DIR}/run_sft_math.sh"
    fi

    # --- Step 4: Evaluate SFT (Arm B) ---
    if [[ "${STEP}" == "all" || "${STEP}" == "eval" ]]; then
        echo ""
        echo "--- Phase 2 Arm B: SFT Evaluation ---"
        python3 "${SRC_DIR}/evaluate_math.py" \
            --model_path "${MODEL_PATH}" \
            --adapter_path "${CKPT_HOME}/math/sft" \
            --output_file "${RESULTS_DIR}/phase2_B_sft.json" \
            --model_name "Arm B: SFT only" \
            --dataset gsm8k
    fi

    # --- Step 5: GRPO from Base (Arm C) ---
    if [[ "${STEP}" == "all" || "${STEP}" == "grpo" ]]; then
        echo ""
        echo "--- Phase 2 Arm C: GRPO from Base ---"
        export DATA_DIR="${DATA_DIR}"
        export CKPT_DIR="${CKPT_HOME}/math/grpo-from-base"
        export EXP_NAME="phase2-grpo-from-base"
        bash "${SRC_DIR}/run_grpo_math.sh"
    fi

    # --- Step 6: Merge SFT checkpoint for Arm D ---
    if [[ "${STEP}" == "all" || "${STEP}" == "grpo" ]]; then
        echo ""
        echo "--- Phase 2: Merging SFT checkpoint for Arm D ---"
        python3 "${SRC_DIR}/merge_lora_checkpoint.py" \
            --base_model "${MODEL_PATH}" \
            --adapter_path "${CKPT_HOME}/math/sft" \
            --output_path "${DATA_HOME}/models/qwen-sft-math-merged"
    fi

    # --- Step 7: GRPO from SFT (Arm D) ---
    if [[ "${STEP}" == "all" || "${STEP}" == "grpo" ]]; then
        echo ""
        echo "--- Phase 2 Arm D: GRPO from SFT ---"
        export MODEL_PATH="${DATA_HOME}/models/qwen-sft-math-merged"
        export DATA_DIR="${DATA_DIR}"
        export CKPT_DIR="${CKPT_HOME}/math/grpo-from-sft"
        export EXP_NAME="phase2-grpo-from-sft"
        bash "${SRC_DIR}/run_grpo_math.sh"
        export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"  # Reset
    fi

    # --- Step 8: Evaluate All Arms ---
    if [[ "${STEP}" == "all" || "${STEP}" == "eval" ]]; then
        echo ""
        echo "--- Phase 2: Batch Evaluation ---"
        python3 "${SRC_DIR}/batch_eval.py" --phase math --results_dir "${RESULTS_DIR}"
    fi
}

# ========================================================================
# DISPATCH
# ========================================================================

case "${PHASE}" in
    1|phase1|multilingual) run_phase1 ;;
    2|phase2|math)         run_phase2 ;;
    all)
        run_phase1
        run_phase2
        ;;
    *)
        echo "Unknown phase: ${PHASE}"
        echo "Usage: $0 [--phase 1|2|all] [--step preprocess|sft|grpo|eval|all]"
        exit 1
        ;;
esac

echo ""
echo "========================================================"
echo "  Pipeline Complete!"
echo "  Results: ${RESULTS_DIR}"
echo "  Time: $(date)"
echo "========================================================"
