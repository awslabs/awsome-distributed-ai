#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --llm ]]; then
    [[ $# == 1 ]] || { printf 'Usage: %s --llm\n' "$0" >&2; exit 2; }
    lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    export LLM_CONFIG=${LLM_CONFIG:-configs/llm.json}
    export LLM_PREP_RECORDS=${LLM_PREP_RECORDS:-160}
    export LLM_MIN_DOCUMENT_TOKENS=${LLM_MIN_DOCUMENT_TOKENS:-4096}
    [[ $LLM_CONFIG == configs/llm.json && -f $lab/$LLM_CONFIG ]] || { printf 'Participant preparation requires configs/llm.json.\n' >&2; exit 2; }
    [[ $LLM_PREP_RECORDS == 160 && $LLM_MIN_DOCUMENT_TOKENS == 4096 ]] || { printf 'Participant preparation requires 160 rows and minimum 4096 tokens.\n' >&2; exit 2; }
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
          PYTHONPATH=/preparation-packages python3 1.prepare-llm.py --config "$LLM_CONFIG" --output /data --records "$LLM_PREP_RECORDS" --minimum-document-tokens "$LLM_MIN_DOCUMENT_TOKENS" --download-weights
          PYTHONPATH=/preparation-packages python3 1.prepare-llm.py --config configs/llm-recovery.json --output /data/recovery --records 1280 --minimum-document-tokens 4096'
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=1 \
        sha256sum "$LLM_DATA_DIR/tokens/tokens.bin" "$LLM_DATA_DIR/tokens/lengths.bin" \
        "$LLM_DATA_DIR/recovery/tokens/tokens.bin" "$LLM_DATA_DIR/recovery/tokens/lengths.bin" \
        "$LLM_DATA_DIR/model/aim347-pin.json"
    exit 0
fi
printf 'Use --llm with explicit assigned paths.\n' >&2
exit 2
