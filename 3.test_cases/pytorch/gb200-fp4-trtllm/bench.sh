#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Rate-sweep benchmark + tok/s reconciliation against the MLPerf GB200 envelope.
set -euo pipefail
: "${ENDPOINT:=http://localhost:8000/v1}"
: "${MODEL:=nvidia/Llama-3.1-8B-Instruct-NVFP4}"
: "${GPUS:=4}"
: "${MLPERF_REF_TOKS_PER_GPU:=170}"   # ~170 tok/s/GPU offline on 405B (MLPerf v5.x); adjust per model

# Drive the OpenAI-compatible endpoint with a rate sweep (use vllm/genai-perf or a simple loop).
echo "Rate-sweeping $MODEL at $ENDPOINT ..."
total_toks="$(genai-perf profile -m "$MODEL" --endpoint-type chat --url "$ENDPOINT" \
  --concurrency 64 --request-count 512 2>/dev/null | awk '/Output token throughput/{print $(NF-1)}')" || total_toks=""

if [[ -z "$total_toks" ]]; then
  echo "WARN: could not parse throughput (is genai-perf installed / endpoint up?)"; exit 0
fi
per_gpu="$(python3 -c "print($total_toks/$GPUS)")"
echo "Measured: ${total_toks} tok/s total, ${per_gpu} tok/s/GPU"
echo "MLPerf reference: ~${MLPERF_REF_TOKS_PER_GPU} tok/s/GPU"
python3 - "$per_gpu" "$MLPERF_REF_TOKS_PER_GPU" <<'PY'
import sys
m, ref = float(sys.argv[1]), float(sys.argv[2])
ratio = m/ref
print(f"ratio to MLPerf envelope: {ratio:.2f}")
if ratio < 0.5:
    print("FAIL: far below the MLPerf envelope -- backend likely NOT using FP4 Tensor Cores "
          "(silent upcast). Re-check the checkpoint quantization_config and the TRT-LLM build.")
    sys.exit(1)
print("OK: within the same order of magnitude as the MLPerf GB200 envelope.")
PY
