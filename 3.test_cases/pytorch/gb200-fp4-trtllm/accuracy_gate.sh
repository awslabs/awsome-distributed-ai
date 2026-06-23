#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Accuracy gate: NVFP4 must stay within ~1% of the BF16/FP8 baseline on MMLU-PRO / GPQA.
set -euo pipefail
: "${MODEL:=nvidia/Llama-3.1-8B-Instruct-NVFP4}"
: "${BASELINE_MMLU_PRO:?set BASELINE_MMLU_PRO to the BF16/FP8 baseline score}"
: "${TASKS:=mmlu_pro,gpqa}"
: "${MAX_DROP:=0.01}"

echo "Evaluating $MODEL on $TASKS ..."
lm_eval --model local-completions \
  --model_args "base_url=http://localhost:8000/v1/completions,model=${MODEL}" \
  --tasks "$TASKS" --output_path /work/lm-eval-nvfp4.json

score="$(python3 -c "import json,glob;d=json.load(open(sorted(glob.glob('/work/lm-eval-nvfp4*.json'))[-1]));print(d['results']['mmlu_pro']['acc,none'])")"
python3 - "$score" "$BASELINE_MMLU_PRO" "$MAX_DROP" <<'PY'
import sys
s, base, maxd = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
drop = base - s
print(f"NVFP4 MMLU-PRO={s:.4f}  baseline={base:.4f}  drop={drop:.4f}  max_allowed={maxd:.4f}")
if drop > maxd:
    print("FAIL: accuracy drop exceeds the PTQ envelope -- re-check calibration / recipe."); sys.exit(1)
print("PASS: NVFP4 accuracy within envelope.")
PY
