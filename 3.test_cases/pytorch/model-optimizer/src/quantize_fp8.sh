#!/usr/bin/env bash
# Run FP8 post-training quantization (PTQ) on a Hugging Face model with NVIDIA Model Optimizer,
# exporting a vLLM/TensorRT-LLM-ready checkpoint.
#
# Defaults to Qwen/Qwen2.5-7B-Instruct (non-gated - no Hugging Face token required).
# Run after setup.sh, as the ec2-user.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EXPORT_PATH="${EXPORT_PATH:-${HOME}/qwen2.5-7b-fp8}"
CALIB_SIZE="${CALIB_SIZE:-512}"
WORKDIR="${HOME}/model-optimizer-recipe"
REPO_DIR="${HOME}/Model-Optimizer"

# shellcheck disable=SC1091
. "${WORKDIR}/modelopt-venv/bin/activate"
cd "${REPO_DIR}/examples/llm_ptq"

# Notes on the flags (each one matters):
#   --qformat fp8            FP8 weights. Works on Ada (L40S) and Hopper (H100). Use nvfp4 on Blackwell.
#   --export_fmt hf          Hugging Face / compressed-tensors checkpoint vLLM can load directly.
#   --dataset cnn_dailymail  Non-gated calibration set. The default Nemotron set is GATED and fails.
#   --kv_cache_qformat none  Quantize WEIGHTS ONLY. FP8 KV-cache is fragile on Ada (garbled output);
#                            enable it only with explicit accuracy validation.
#   --attn_implementation eager  Avoids a slow flash-attn source build; fine for calibration.
python hf_ptq.py \
  --pyt_ckpt_path "${MODEL}" \
  --qformat fp8 \
  --export_fmt hf \
  --dataset cnn_dailymail \
  --calib_size "${CALIB_SIZE}" \
  --batch_size 4 \
  --kv_cache_qformat none \
  --attn_implementation eager \
  --trust_remote_code \
  --export_path "${EXPORT_PATH}"

echo "==> FP8 checkpoint exported to: ${EXPORT_PATH}"
du -sh "${EXPORT_PATH}" || true
echo "==> next: python smoke_test_vllm.py --model ${EXPORT_PATH}"
