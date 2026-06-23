#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Serve a model on GB200 with KV-cache offload to Grace host memory (cold tier).
set -euo pipefail
: "${MODEL:=deepseek-ai/DeepSeek-V3}"
: "${TP:=4}"
: "${CPU_OFFLOAD_GB:=320}"     # fraction of the ~480 GB Grace pool; leave OS/co-tenant headroom
: "${PORT:=8000}"

echo "GB200 KV-offload :: $MODEL  TP=$TP  Grace cold-KV pool=${CPU_OFFLOAD_GB} GiB"
echo "NOTE: Grace is ~1/16 HBM bandwidth -- this buys CAPACITY (longer context / more"
echo "      concurrency), not decode speed. Active KV stays in HBM."

# vLLM host-KV offload. (Add LMCache via --kv-transfer-config for tiered HBM->Grace->NVMe.)
exec vllm serve "$MODEL" \
  --tensor-parallel-size "$TP" \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" \
  --host 0.0.0.0 --port "$PORT"
  # Optional GPU-resident speculative decoding (NOT a Grace-CPU draft):
  #   --speculative-config '{"method":"eagle3","model":"<eagle3-head>","num_speculative_tokens":3}'
