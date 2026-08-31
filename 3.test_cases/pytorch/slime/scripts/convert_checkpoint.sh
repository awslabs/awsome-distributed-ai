#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# ============================================================
# Checkpoint Conversion Helper
#
# Converts between HuggingFace and Megatron torch_dist formats.
#
# Usage:
#   # HuggingFace -> Megatron (required before training)
#   bash scripts/convert_checkpoint.sh hf2megatron \
#       --model-script qwen3-4B.sh \
#       --hf-path /fsx/models/Qwen3-4B \
#       --save-path /fsx/models/Qwen3-4B_torch_dist
#
#   # Megatron -> HuggingFace (after training, for evaluation)
#   bash scripts/convert_checkpoint.sh megatron2hf \
#       --input-dir /fsx/checkpoints/qwen3-4b-grpo/iter_0060/ \
#       --output-dir /fsx/models/Qwen3-4B-GRPO-step60 \
#       --origin-hf-dir /fsx/models/Qwen3-4B
# ============================================================

set -euo pipefail

DIRECTION="${1:-}"
shift || true

if [[ -z "${DIRECTION}" ]]; then
    echo "Usage: $0 {hf2megatron|megatron2hf} [options]"
    exit 1
fi

SLIME_DIR="${SLIME_DIR:-/opt/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-/opt/Megatron-LM}"

case "${DIRECTION}" in
    hf2megatron)
        MODEL_SCRIPT=""
        HF_PATH=""
        SAVE_PATH=""
        NUM_GPUS=1

        while [[ $# -gt 0 ]]; do
            case "$1" in
                --model-script) MODEL_SCRIPT="$2"; shift 2 ;;
                --hf-path) HF_PATH="$2"; shift 2 ;;
                --save-path) SAVE_PATH="$2"; shift 2 ;;
                --num-gpus) NUM_GPUS="$2"; shift 2 ;;
                *) echo "Unknown option: $1"; exit 1 ;;
            esac
        done

        if [[ -z "${MODEL_SCRIPT}" || -z "${HF_PATH}" || -z "${SAVE_PATH}" ]]; then
            echo "Required: --model-script, --hf-path, --save-path"
            exit 1
        fi

        echo "[INFO] Converting HuggingFace -> Megatron torch_dist"
        echo "  Model script: ${MODEL_SCRIPT}"
        echo "  HF path:      ${HF_PATH}"
        echo "  Save path:    ${SAVE_PATH}"
        echo "  GPUs:         ${NUM_GPUS}"

        cd "${SLIME_DIR}"
        source "scripts/models/${MODEL_SCRIPT}"

        if [[ ${NUM_GPUS} -gt 1 ]]; then
            PYTHONPATH="${MEGATRON_DIR}" torchrun \
                --nproc_per_node="${NUM_GPUS}" \
                tools/convert_hf_to_torch_dist.py \
                "${MODEL_ARGS[@]}" \
                --hf-checkpoint "${HF_PATH}" \
                --save "${SAVE_PATH}"
        else
            PYTHONPATH="${MEGATRON_DIR}" python3 \
                tools/convert_hf_to_torch_dist.py \
                "${MODEL_ARGS[@]}" \
                --hf-checkpoint "${HF_PATH}" \
                --save "${SAVE_PATH}"
        fi

        echo "[INFO] Conversion complete: ${SAVE_PATH}"
        ;;

    megatron2hf)
        INPUT_DIR=""
        OUTPUT_DIR=""
        ORIGIN_HF_DIR=""

        while [[ $# -gt 0 ]]; do
            case "$1" in
                --input-dir) INPUT_DIR="$2"; shift 2 ;;
                --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
                --origin-hf-dir) ORIGIN_HF_DIR="$2"; shift 2 ;;
                *) echo "Unknown option: $1"; exit 1 ;;
            esac
        done

        if [[ -z "${INPUT_DIR}" || -z "${OUTPUT_DIR}" || -z "${ORIGIN_HF_DIR}" ]]; then
            echo "Required: --input-dir, --output-dir, --origin-hf-dir"
            exit 1
        fi

        echo "[INFO] Converting Megatron torch_dist -> HuggingFace"
        echo "  Input:     ${INPUT_DIR}"
        echo "  Output:    ${OUTPUT_DIR}"
        echo "  Origin HF: ${ORIGIN_HF_DIR}"

        cd "${SLIME_DIR}"

        PYTHONPATH="${MEGATRON_DIR}" python3 \
            tools/convert_torch_dist_to_hf.py \
            --input-dir "${INPUT_DIR}" \
            --output-dir "${OUTPUT_DIR}" \
            --origin-hf-dir "${ORIGIN_HF_DIR}"

        echo "[INFO] Conversion complete: ${OUTPUT_DIR}"
        ;;

    *)
        echo "Unknown direction: ${DIRECTION}"
        echo "Usage: $0 {hf2megatron|megatron2hf} [options]"
        exit 1
        ;;
esac
