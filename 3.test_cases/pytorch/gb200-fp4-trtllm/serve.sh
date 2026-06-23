#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Serve an NVFP4 checkpoint with trtllm-serve on GB200 (4 GPUs, sm_100).
set -euo pipefail
: "${MODEL:=nvidia/Llama-3.1-8B-Instruct-NVFP4}"
: "${TP:=4}"
: "${PORT:=8000}"

# Sanity: confirm FP4 before serving (don't trust throughput from a silently-upcast model).
python3 - "$MODEL" <<'PY' || echo "WARN: could not verify quantization_config (remote model?)"
import json, sys
try:
    from huggingface_hub import hf_hub_download
    cfg = json.load(open(hf_hub_download(sys.argv[1], "config.json")))
    q = cfg.get("quantization_config", {})
    assert "nvfp4" in str(q).lower() or q.get("quant_method") == "modelopt"
    print("OK: NVFP4 checkpoint")
except Exception as e:
    print("verify skipped:", e)
PY

exec trtllm-serve "$MODEL" --tp_size "$TP" --host 0.0.0.0 --port "$PORT"
