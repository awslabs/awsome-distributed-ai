#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# Model Download Script for Qwen/Qwen2.5-Coder-7B-Instruct
# Downloads model weights to FSx shared storage for fast local loading
# Eliminates per-worker HuggingFace Hub downloads at training startup
# =============================================================================
set -xeuo pipefail

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/../env_vars" ]; then
    source "${SCRIPT_DIR}/../env_vars"
fi

# Configuration
RAY_DATA_HOME=${RAY_DATA_HOME:-"/fsx/data/verl"}
HF_MODEL_ID="Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_NAME="Qwen2.5-Coder-7B-Instruct"
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
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${HF_MODEL_ID}', local_dir='${MODEL_DIR}')
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

model_dir = os.environ.get("MODEL_DIR", "")
shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))

if not shards:
    print(f"ERROR: No .safetensors files found in {model_dir}")
    sys.exit(1)

try:
    from safetensors import safe_open
except ImportError:
    print("WARNING: safetensors package not installed, skipping integrity check")
    sys.exit(0)

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
echo ""
echo "The training scripts will auto-detect this local copy."
echo "To use it manually, set in env_vars:"
echo "  export MODEL_PATH=\"${MODEL_DIR}\""
