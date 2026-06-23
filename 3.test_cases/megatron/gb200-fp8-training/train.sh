#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Precision-knob Megatron-LM launcher for GB200. Translate PRECISION -> Megatron/TE flags
# and keep TP/EP within the NVLink domain. Invoked by srun/mpirun (rank env provided).
set -euo pipefail

: "${PRECISION:=fp8-mxfp8}"     # bf16 | fp8-mxfp8 | fp8-tensorwise | fp8-delayed | fp4-nvfp4
: "${MODEL:=llama3-8b}"
: "${TP:=4}"                    # tensor parallel -- keep <= NVLink domain (<=72)
: "${PP:=1}"                    # pipeline parallel -- across EFA between UltraServers
: "${MICRO_BS:=1}"
: "${GLOBAL_BS:=512}"
: "${SEQ_LEN:=8192}"
: "${TRAIN_ITERS:=200}"

# Canonical GB200 NCCL+EFA env (NVLS intra-domain, NVLSTree over EFA cross-domain).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../../../micro-benchmarks/nccl-tests/gb200-env.sh" 2>/dev/null || \
  source /opt/megatron-gb200/gb200-env.sh 2>/dev/null || true

case "$PRECISION" in
  bf16)           PREC_ARGS=(--bf16) ;;
  fp8-mxfp8)      PREC_ARGS=(--bf16 --fp8-format hybrid --fp8-recipe mxfp8) ;;
  fp8-tensorwise) PREC_ARGS=(--bf16 --fp8-format hybrid --fp8-recipe tensorwise) ;;
  fp8-delayed)    PREC_ARGS=(--bf16 --fp8-format hybrid --fp8-recipe delayed) ;;
  fp4-nvfp4)      PREC_ARGS=(--bf16 --fp4 nvfp4)   # EVAL/throughput only -- FP4 training is maturing
                  echo "WARNING: fp4-nvfp4 is for throughput/eval, not converged training." >&2 ;;
  *) echo "Unknown PRECISION=$PRECISION" >&2; exit 2 ;;
esac

declare -a ARGS=(
  --tensor-model-parallel-size "$TP"
  --pipeline-model-parallel-size "$PP"
  --seq-length "$SEQ_LEN"
  --micro-batch-size "$MICRO_BS"
  --global-batch-size "$GLOBAL_BS"
  --train-iters "$TRAIN_ITERS"
  --transformer-impl transformer_engine
  --use-mcore-models
  "${PREC_ARGS[@]}"
)

echo "GB200 Megatron training :: MODEL=$MODEL PRECISION=$PRECISION TP=$TP PP=$PP"
echo "Args: ${ARGS[*]}"
# Hand off to Megatron's pretrain entrypoint (model-specific args appended via MODEL config).
exec python -u pretrain_gpt.py "${ARGS[@]}" ${MEGATRON_EXTRA_ARGS:-}
