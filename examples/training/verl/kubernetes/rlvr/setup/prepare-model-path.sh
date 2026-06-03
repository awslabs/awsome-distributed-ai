#!/bin/bash
set -euo pipefail

# Prepare a local model path for the Qwen3-VL-235B-A22B-Instruct checkpoint.
# This creates the target directory under ${RAY_DATA_HOME}/models and prints
# the HF_MODEL_PATH you can use in jobs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_VARS_FILE="${SCRIPT_DIR}/env_vars"

if [ ! -f "${ENV_VARS_FILE}" ]; then
  echo "Missing ${ENV_VARS_FILE}. Copy env_vars.example and set your values."
  exit 1
fi

source "${ENV_VARS_FILE}"

RAY_DATA_HOME="${RAY_DATA_HOME:-/fsx/verl}"
MODEL_LOCAL_NAME="${MODEL_LOCAL_NAME:-Qwen3-VL-235B-A22B-Instruct}"
TARGET_PATH="${RAY_DATA_HOME}/models/${MODEL_LOCAL_NAME}"

mkdir -p "${TARGET_PATH}"

cat > "${TARGET_PATH}/README.txt" <<'NOTE'
Place the local model files for Qwen3-VL-235B-A22B-Instruct here.
If you are using Hugging Face Hub, set MODEL_PATH (or HF_MODEL_PATH) to a valid
Hub repo ID instead, e.g. "Qwen/Qwen2-VL-7B-Instruct", and skip local files.
NOTE

echo "Prepared model directory: ${TARGET_PATH}"
echo "Set HF_MODEL_PATH (or MODEL_PATH) to: ${TARGET_PATH}"

