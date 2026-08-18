#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# Model Download Script (Ray Job Compatible)
#
# Downloads HuggingFace model weights to FSx shared storage.
# Designed to run as a Ray job so it executes on a cluster node with
# FSx mounted, using the base image's huggingface_hub — no extra
# pip installs needed via runtime_env.
#
# Usage (via Ray job submit):
#   ray job submit --address $RAY_ADDRESS --no-wait \
#     --runtime-env-json '{"env_vars": {"HF_TOKEN": "..."}}' \
#     --entrypoint-num-cpus 4 \
#     -- python models/download_model.py --model Qwen/Qwen3-8B
#
# Usage (local, for testing):
#   python models/download_model.py --model Qwen/Qwen3-8B --output-dir /tmp/models
# =============================================================================

import argparse
import glob
import os
import sys
import time

# Well-known models and their recommended output names
MODEL_ALIASES = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-coder-next": "Qwen/Qwen3-Coder-Next",
    "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen3-235b": "Qwen/Qwen3-235B-A22B",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
}


def download_model(model_id, output_dir, hf_token=None):
    """Download model from HuggingFace Hub to local directory."""
    from huggingface_hub import snapshot_download

    model_name = model_id.split("/")[-1]
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print(f"Downloading: {model_id}")
    print(f"Output:      {model_dir}")
    print("=" * 60)

    t0 = time.time()
    snapshot_download(
        repo_id=model_id,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        token=hf_token,
    )
    elapsed = time.time() - t0
    print(f"\nDownload completed in {elapsed:.0f}s")

    # Verify config.json exists
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"ERROR: config.json not found in {model_dir}")
        sys.exit(1)

    return model_dir


def verify_safetensors(model_dir):
    """Verify integrity of all safetensor shards."""
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        print(f"WARNING: No .safetensors files found in {model_dir}")
        return True

    try:
        from safetensors import safe_open
    except ImportError:
        print("WARNING: safetensors package not installed, skipping integrity check")
        return True

    print(f"\nVerifying {len(shards)} safetensor shards...")
    corrupt = []
    total_tensors = 0
    total_bytes = 0
    for path in shards:
        name = os.path.basename(path)
        size_bytes = os.path.getsize(path)
        try:
            with safe_open(path, framework="pt") as f:
                n_tensors = len(f.keys())
            total_tensors += n_tensors
            total_bytes += size_bytes
            print(f"  OK: {name} ({n_tensors} tensors, {size_bytes / 1e9:.2f} GB)")
        except Exception as e:
            print(f"  CORRUPT: {name} ({size_bytes / 1e9:.2f} GB) -> {e}")
            corrupt.append(name)

    if corrupt:
        print(f"\nERROR: {len(corrupt)} of {len(shards)} shards are corrupt:")
        for name in corrupt:
            print(f"  - {name}")
        return False

    print(f"\nAll {len(shards)} shards verified OK")
    print(f"  Total tensors: {total_tensors}")
    print(f"  Total size: {total_bytes / 1e9:.1f} GB")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace model to FSx")
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID (e.g., Qwen/Qwen3-8B) or alias (e.g., qwen3-8b)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Base output directory (default: /fsx/data/verl/models)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip safetensor shard verification",
    )
    args = parser.parse_args()

    # Resolve alias
    model_id = MODEL_ALIASES.get(args.model.lower(), args.model)

    # Default output dir
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = "/fsx/data/verl/models"

    # HF token from env
    hf_token = os.environ.get("HF_TOKEN")

    # Download
    model_dir = download_model(model_id, output_dir, hf_token)

    # Verify
    if not args.skip_verify:
        if not verify_safetensors(model_dir):
            print("\nRe-run with --skip-verify to skip, or re-download.")
            sys.exit(1)

    # Summary
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Path:  {model_dir}")
    print("\nTo use in training, set:")
    print(f'  export MODEL_PATH="{model_dir}"')


if __name__ == "__main__":
    main()
