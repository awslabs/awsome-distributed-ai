#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/lib/common.sh"
: "${DATA_DIR:?Set DATA_DIR}" "${LAB_IMAGE:?Set LAB_IMAGE to an absolute .sqsh path}"
cd "$LAB_DIR"
docker build -t aim347-lab:2026-09-06 .
# This import uses the local Docker daemon and does not publish an image.
enroot import -o "$LAB_IMAGE" dockerd://aim347-lab:2026-09-06
mkdir -p "$DATA_DIR/tokens"
docker run --rm --user "$(id -u):$(id -g)" -v "$DATA_DIR:/data" aim347-lab:2026-09-06 \
  python3 lib/data.py /data/tokens
# Stage a pretrained model of the same architecture for the serving pivot.
docker run --rm --user "$(id -u):$(id -g)" -e HF_HOME=/data/hf-cache -v "$DATA_DIR:/data" \
  aim347-lab:2026-09-06 python3 -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen2.5-3B", revision="3aab1f1954e9cc14eb9509a215f9e5ca08227a9b", local_dir="/data/serving-model", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])'
enroot import -o "${VLLM_IMAGE:?Set VLLM_IMAGE to an absolute .sqsh path}" docker://vllm/vllm-openai:v0.20.2
