#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# GRPO RL post-training on GB200 with slime: BF16 train + FP8 SGLang rollout, EP/TP inside
# the 72-GPU NVLink domain. Ray head/workers must be pinned within one clique.
set -euo pipefail
source /opt/slime/gb200-env.sh 2>/dev/null || \
  source ../../../micro-benchmarks/nccl-tests/gb200-env.sh 2>/dev/null || true

: "${MODEL:=Qwen/Qwen3-30B-A3B}"   # MoE showcase; or DeepSeek-V3-class
: "${DATASET:=dapo-math-17k}"
: "${TP:=8}"
: "${EP:=8}"                       # keep <= 72 (one NVLink domain)
: "${MAX_STEPS:=500}"

if (( EP > 72 )); then echo "EP=$EP exceeds the 72-GPU NVLink domain" >&2; exit 2; fi

echo "GB200 slime GRPO :: $MODEL  TP=$TP EP=$EP  (BF16 train / FP8 rollout)"
exec python -m slime.train \
  --algorithm grpo \
  --model "$MODEL" --dataset "$DATASET" \
  --train-precision bf16 \
  --sglang-kv-cache-dtype fp8_e4m3 \
  --tensor-parallel "$TP" --expert-parallel "$EP" \
  --max-steps "$MAX_STEPS" \
  ${SLIME_EXTRA_ARGS:-}
