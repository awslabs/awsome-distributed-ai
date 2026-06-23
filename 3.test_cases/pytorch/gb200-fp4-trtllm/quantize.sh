#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# BYO NVFP4 quantization via NVIDIA ModelOpt. Skip this if you pull a pre-quantized
# nvidia/*-NVFP4 checkpoint (the fast path).
set -euo pipefail
: "${HF_MODEL:?set HF_MODEL to the source HF model id or path}"
: "${TP:=4}"
: "${OUT:=/work/nvfp4-ckpt}"
: "${CALIB_SAMPLES:=256}"          # 128-512 is plenty for NVFP4 PTQ

# ModelOpt's HF example handles calibration + export to an NVFP4 checkpoint.
scripts/huggingface_example.sh \
  --model "$HF_MODEL" \
  --quant nvfp4 \
  --tp "$TP" \
  --calib "$CALIB_SAMPLES" \
  --export_path "$OUT"

echo "NVFP4 checkpoint at $OUT"
python3 - "$OUT" <<'PY'
import json, sys, os
cfg = os.path.join(sys.argv[1], "config.json")
q = json.load(open(cfg)).get("quantization_config", {})
print("quantization_config:", q)
assert "nvfp4" in str(q).lower() or q.get("quant_method") == "modelopt", \
    "FAIL: exported checkpoint is not NVFP4/modelopt"
print("OK: checkpoint is NVFP4 (modelopt)")
PY
