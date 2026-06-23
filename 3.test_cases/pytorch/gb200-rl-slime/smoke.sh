#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Single-node GB200 smoke run: Qwen3-4B on DAPO-math-17k. Run this BEFORE NVL72 scale to
# reproduce-or-clear the known slime #1487 hang on the Grace-Blackwell container path.
set -euo pipefail
source /opt/slime/gb200-env.sh 2>/dev/null || \
  source ../../../micro-benchmarks/nccl-tests/gb200-env.sh 2>/dev/null || true

: "${MODEL:=Qwen/Qwen3-4B}"
: "${DATASET:=dapo-math-17k}"

echo "GB200 slime smoke :: $MODEL on $DATASET (single node, 4 GPUs)"
echo "If this hangs at rollout init, that is #1487 -- check the Grace-Blackwell NCCL fixes"
echo "before scaling out."

exec python -m slime.train \
  --model "$MODEL" --dataset "$DATASET" \
  --train-precision bf16 \
  --sglang-kv-cache-dtype fp8_e4m3 \
  --tensor-parallel 4 \
  --max-steps 5
