#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/lib/common.sh"
: "${DATA_DIR:?Set DATA_DIR}" "${LAB_IMAGE:?Set LAB_IMAGE to an absolute .sqsh path}"
cd "$LAB_DIR"
if [[ -n ${PREBUILT_LAB_IMAGE:-} ]]; then
    [[ $PREBUILT_LAB_IMAGE == *@sha256:* ]] || { echo 'PREBUILT_LAB_IMAGE must use an immutable digest' >&2; exit 2; }
    docker pull "$PREBUILT_LAB_IMAGE"
    docker tag "$PREBUILT_LAB_IMAGE" aim347-lab:2026-09-06
else
    docker build --build-arg NCCL_BUILD_JOBS="${NCCL_BUILD_JOBS:-8}" -t aim347-lab:2026-09-06 .
fi
# This import uses the local Docker daemon and does not publish an image.
if [[ ! -e $LAB_IMAGE ]]; then
    enroot import -o "$LAB_IMAGE" dockerd://aim347-lab:2026-09-06
fi
unsquashfs -s "$LAB_IMAGE"
mkdir -p "$DATA_DIR/tokens"
if [[ ! -s $DATA_DIR/tokens/manifest.json ]]; then
    docker run --rm --user "$(id -u):$(id -g)" -v "$DATA_DIR:/data" aim347-lab:2026-09-06 \
      python3 lib/data.py /data/tokens
fi
cat "$DATA_DIR/tokens/manifest.json"
# Stage a pretrained model of the same architecture for the serving pivot.
docker run --rm --user "$(id -u):$(id -g)" -e HF_HOME=/data/hf-cache -v "$DATA_DIR:/data" \
  aim347-lab:2026-09-06 python3 -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen2.5-3B", revision="3aab1f1954e9cc14eb9509a215f9e5ca08227a9b", local_dir="/data/serving-model", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])'
: "${VLLM_IMAGE:?Set VLLM_IMAGE to an absolute .sqsh path}"
if [[ ! -e $VLLM_IMAGE ]]; then
    # Direct registry import of this platform digest failed with Enroot 3.5.
    # Resolve the immutable digest with Docker before the local import.
    docker pull vllm/vllm-openai@sha256:68b773151407ca28c05479c02c0c08a573285ba197c8ff35d2103acd97aff78c
    docker tag vllm/vllm-openai@sha256:68b773151407ca28c05479c02c0c08a573285ba197c8ff35d2103acd97aff78c aim347-vllm:v0.20.2
    enroot import -o "$VLLM_IMAGE" dockerd://aim347-vllm:v0.20.2
fi
unsquashfs -s "$VLLM_IMAGE"
