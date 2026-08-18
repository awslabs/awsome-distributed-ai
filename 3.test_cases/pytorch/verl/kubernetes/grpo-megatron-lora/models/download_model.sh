#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# Model Download Script (parameterised)
# Downloads model weights to FSx shared storage for fast local loading, which
# eliminates per-worker HuggingFace Hub downloads at training startup.
#
# This runs INSIDE the fsx-utils pod, dispatched by models/run_on_cluster.sh --
# it needs /fsx mounted and does not need a running Ray cluster. For the Ray-job
# path instead, use models/submit_download.sh <HF_ID> <DEST>.
#
# Usage:
#   ./models/run_on_cluster.sh models/download_model.sh <HF_MODEL_ID> [MODEL_NAME]
#
# Examples:
#   ./models/run_on_cluster.sh models/download_model.sh Qwen/Qwen3-8B
#   ./models/run_on_cluster.sh models/download_model.sh Qwen/Qwen3-235B-A22B
#
# MODEL_NAME defaults to the repo name (the part after "/") and is the directory
# created under ${RAY_DATA_HOME}/models/. It must match conf/model/*.yaml `name`,
# because fsx_path is ${compute.fsx_home}/models/${model.name}.
#
# Model IDs for the shipped conf/model/ groups are listed in
# docs/configuration.md ("Model download reference").
# =============================================================================
# NOTE: deliberately NOT `set -x`. Bash xtrace applies to commands executed inside
# a sourced file, so tracing here would echo every `export` in env_vars --
# including HF_TOKEN -- to stderr, and run_on_cluster.sh runs this via kubectl
# exec, so that lands in pod logs.
set -euo pipefail

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/../env_vars" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/../env_vars"
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <HF_MODEL_ID> [MODEL_NAME]" >&2
    echo "Example: $0 Qwen/Qwen3-8B" >&2
    exit 1
fi

# Configuration
RAY_DATA_HOME=${RAY_DATA_HOME:-"/fsx/data/verl"}
HF_MODEL_ID="$1"
MODEL_NAME="${2:-${HF_MODEL_ID##*/}}"
MODEL_DIR="${RAY_DATA_HOME}/models/${MODEL_NAME}"

# Create model directory
mkdir -p "${MODEL_DIR}"

echo "=============================================="
echo "Downloading ${HF_MODEL_ID}"
echo "=============================================="
echo "Model:            ${HF_MODEL_ID}"
echo "Output directory: ${MODEL_DIR}"
echo "=============================================="

# Download model using huggingface_hub Python API
# snapshot_download creates a self-contained copy (no symlinks to HF cache)
# This is required for verl/vLLM to load from a local path correctly
HF_MODEL_ID="${HF_MODEL_ID}" MODEL_DIR="${MODEL_DIR}" python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ['HF_MODEL_ID'], local_dir=os.environ['MODEL_DIR'])
"

# Verify download -- config.json must exist
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "ERROR: Download failed -- config.json not found in ${MODEL_DIR}"
    exit 1
fi

# Verify safetensor shard integrity
# A partial download or FSx sync issue can leave truncated .safetensors files that
# pass the config.json check but fail at training time with:
#   SafetensorError: Error while deserializing header: incomplete metadata
echo ""
echo "Verifying safetensor shard integrity..."
MODEL_DIR="${MODEL_DIR}" python3 << 'VERIFY_EOF'
import glob, os, sys

from safetensors import safe_open

model_dir = os.environ.get("MODEL_DIR", "")
shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))

if not shards:
    print(f"ERROR: No .safetensors files found in {model_dir}")
    sys.exit(1)

corrupt = []
for path in shards:
    name = os.path.basename(path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    try:
        with safe_open(path, framework="pt") as f:
            n_tensors = len(f.keys())
        print(f"  OK: {name} ({n_tensors} tensors, {size_mb:.1f} MB)")
    except Exception as e:
        print(f"  CORRUPT: {name} ({size_mb:.1f} MB) -> {e}")
        corrupt.append(name)

if corrupt:
    print(f"\nERROR: {len(corrupt)} of {len(shards)} safetensor shards are corrupt:")
    for name in corrupt:
        print(f"  - {name}")
    print("\nRe-run this script to retry the download")
    sys.exit(1)

print(f"\nAll {len(shards)} safetensor shards verified OK")
VERIFY_EOF

# Show downloaded files
echo ""
echo "=============================================="
echo "Download complete!"
echo "=============================================="
ls -lh "${MODEL_DIR}/"
echo ""
echo "Model saved to: ${MODEL_DIR}"
