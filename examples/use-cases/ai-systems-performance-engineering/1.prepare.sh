#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --llm ]]; then
    [[ $# == 1 ]] || { printf 'Usage: %s --llm\n' "$0" >&2; exit 2; }
    lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    export LLM_CONFIG=${LLM_CONFIG:-configs/llm.json}
    export LLM_PREP_RECORDS=${LLM_PREP_RECORDS:-1024}
    export LLM_MIN_DOCUMENT_TOKENS=${LLM_MIN_DOCUMENT_TOKENS:-0}
    [[ $LLM_CONFIG == configs/*.json && $LLM_CONFIG != *..* && -f $lab/$LLM_CONFIG ]] || exit 2
    [[ $LLM_PREP_RECORDS =~ ^[1-9][0-9]*$ && $LLM_MIN_DOCUMENT_TOKENS =~ ^[0-9]+$ ]] || exit 2
    : "${SLURM_JOB_ID:?Run preparation in the assigned Slurm allocation}"
    : "${SLURM_JOB_NUM_NODES:?}" "${LAB_IMAGE:?}" "${VLLM_IMAGE:?}"
    : "${LLM_DATA_DIR:?}" "${LLM_PREPARATION_PACKAGES:?}"
    : "${LAB_IMAGE_SHA256:?}" "${VLLM_IMAGE_SHA256:?}"
    [[ $SLURM_JOB_NUM_NODES == 2 ]] || { printf 'Prepare both assigned nodes.\n' >&2; exit 2; }
    : "${ASSIGNED_NODES:?Set the exact assigned compute pair}"
    export ASSIGNED_NODES
    python3 - <<'PY'
import os, subprocess
assigned = os.environ['ASSIGNED_NODES'].split(',')
allocated = subprocess.check_output(
    ['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']], text=True).split()
if len(assigned) != 2 or len(set(assigned)) != 2 or sorted(assigned) != sorted(allocated):
    raise ValueError('preparation allocation differs from the assigned pair')
PY
    for path in "$lab" "$LLM_DATA_DIR" "$LLM_PREPARATION_PACKAGES" "$LAB_IMAGE" "$VLLM_IMAGE"; do
        [[ $path == /* && $path != *[[:space:],:]* ]] || { printf 'Use absolute mount-safe paths.\n' >&2; exit 2; }
    done
    [[ $(findmnt -T "$(dirname -- "$LLM_DATA_DIR")" -n -o FSTYPE) == lustre ]]
    # Check actual images on both nodes. No replacement of retained images.
    # shellcheck disable=SC2016
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=1 \
        bash -c 'set -eu; unsquashfs -s "$1"; unsquashfs -s "$2"; printf "%s  %s\n%s  %s\n" "$3" "$1" "$4" "$2" | sha256sum --check --strict' \
        bash "$LAB_IMAGE" "$VLLM_IMAGE" "$LAB_IMAGE_SHA256" "$VLLM_IMAGE_SHA256"
    # The data output is fresh. A failed partial preparation remains for diagnosis.
    mkdir "$LLM_DATA_DIR"
    preparation_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    srun --nodes=1 --ntasks=1 --nodelist="$preparation_node" --cpus-per-task=1 mkdir "$LLM_PREPARATION_PACKAGES"
    srun --nodes=1 --ntasks=1 --nodelist="$preparation_node" --cpus-per-task=8 \
        --container-image="$LAB_IMAGE" \
        --container-mounts="$lab:/opt/aim347,$LLM_DATA_DIR:/data,$LLM_PREPARATION_PACKAGES:/preparation-packages" \
        --container-workdir=/opt/aim347 --no-container-remap-root \
        --container-env=LLM_CONFIG,LLM_PREP_RECORDS,LLM_MIN_DOCUMENT_TOKENS \
        bash -c 'set -euo pipefail
          python3 -m pip install --disable-pip-version-check --target /preparation-packages -r requirements-data.txt -c constraints-data.txt
          python3 -m pip list --path /preparation-packages --format=json > /data/preparation-packages.json
          PYTHONPATH=/preparation-packages python3 1.prepare-llm.py --config "$LLM_CONFIG" --output /data --records "$LLM_PREP_RECORDS" --minimum-document-tokens "$LLM_MIN_DOCUMENT_TOKENS" --download-weights'
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=1 \
        sha256sum "$LLM_DATA_DIR/tokens/tokens.bin" "$LLM_DATA_DIR/tokens/lengths.bin" "$LLM_DATA_DIR/model/aim347-pin.json"
    exit 0
fi
[[ $# == 0 ]] || { printf 'Usage: %s [--llm]\n' "$0" >&2; exit 2; }
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
