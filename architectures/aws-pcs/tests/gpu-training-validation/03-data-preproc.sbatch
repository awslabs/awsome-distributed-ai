#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Stage 3b: Preprocess a small sample corpus into Megatron's indexed format,
# using the Llama2 tokenizer. Adapted from
# 3.test_cases/megatron/megatron-lm/slurm/llama2/data-preproc-llama2.sbatch.
#
# Prereqs (place under ${DATA_PATH}/llama2 before running):
#   - tokenizer.model  (from huggingface.co/meta-llama/Llama-2-7b-hf, gated)
#   - oscar-1GB.jsonl  (decompressed sample corpus)

#SBATCH --job-name=megatron-data-preproc
#SBATCH --partition=gpu-p5
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=03-data-preproc_%j.out
#SBATCH --error=03-data-preproc_%j.out

set -euxo pipefail

# DATA_PATH is the path *inside* the container; we bind the validation dir to it.
: "${IMAGE:=/fsx/validation/aws-megatron-lm.sqsh}"
: "${VALIDATION_DIR:=/fsx/validation}"
: "${DATA_PATH:=/fsx}"
: "${FSX_MOUNT:=${VALIDATION_DIR}:${DATA_PATH}}"

[[ -f "${IMAGE}" ]] || { echo "Missing ${IMAGE}; run 03-build-megatron.sbatch first." >&2; exit 1; }
[[ -f "${VALIDATION_DIR}/llama2/tokenizer.model" ]] || {
  echo "Missing ${VALIDATION_DIR}/llama2/tokenizer.model (download from HF, gated)." >&2; exit 1; }
[[ -f "${VALIDATION_DIR}/llama2/oscar-1GB.jsonl" ]] || {
  echo "Missing ${VALIDATION_DIR}/llama2/oscar-1GB.jsonl (wget + xz -d)." >&2; exit 1; }

declare -a ARGS=(
    --container-image "${IMAGE}"
    --container-mount-home
    --container-mounts "${FSX_MOUNT}"
)

srun -l "${ARGS[@]}" python3 /workspace/Megatron-LM/tools/preprocess_data.py \
    --input "${DATA_PATH}/llama2/oscar-1GB.jsonl" \
    --output-prefix "${DATA_PATH}/llama2/my-llama2" \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model "${DATA_PATH}/llama2/tokenizer.model" \
    --append-eod \
    --workers 64
