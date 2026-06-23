#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Short Megatron-Core MoE run on GB200, comparing the flex dispatcher backends.
# BACKEND=deepep   -> DeepEP v1 (NVSHMEM libfabric on EFA)
# BACKEND=hybridep -> MNNVL-native hybrid-ep (the GB200 path for the 72-GPU domain)
set -euo pipefail

: "${BACKEND:=hybridep}"     # deepep | hybridep
: "${EP:=8}"                 # expert parallel -- keep <= 72 (one NVLink domain)
: "${TP:=4}"
: "${MODEL:=mixtral-8x7b}"   # or a small DeepSeek-V3-shaped config
: "${TRAIN_ITERS:=100}"

source /opt/megatron-gb200/gb200-env.sh 2>/dev/null || \
  source ../../../micro-benchmarks/nccl-tests/gb200-env.sh 2>/dev/null || true

if (( EP > 72 )); then
  echo "ERROR: EP=$EP exceeds the 72-GPU NVLink domain. Keep EP<=72 and scale with PP/DP." >&2
  exit 2
fi

echo "GB200 MoE training :: MODEL=$MODEL BACKEND=$BACKEND EP=$EP TP=$TP"
exec python -u pretrain_gpt.py \
  --num-experts 8 --moe-router-topk 2 \
  --expert-model-parallel-size "$EP" \
  --tensor-model-parallel-size "$TP" \
  --moe-token-dispatcher-type flex \
  --moe-flex-dispatcher-backend "$BACKEND" \
  --transformer-impl transformer_engine \
  --bf16 --fp8-format hybrid --fp8-recipe mxfp8 \
  --train-iters "$TRAIN_ITERS" \
  ${MEGATRON_EXTRA_ARGS:-}
