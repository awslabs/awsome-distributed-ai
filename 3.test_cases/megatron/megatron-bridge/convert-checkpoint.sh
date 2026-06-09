#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# convert-checkpoint.sh  (shared, model-agnostic)
#
# Download a HuggingFace model's weights, dequantize block-FP8 -> BF16 (inline
# during import), and convert to a Megatron-Core distributed checkpoint on FSx
# via Megatron-Bridge AutoBridge.import_ckpt().
#
# This is the library-level checkpoint-conversion step, shared by every model
# under megatron-bridge/. It bakes in NO model: the model, revision, FSx root,
# and trust_remote_code flag are all supplied via environment variables, so one
# script serves Kimi-K2, DeepSeek-V3, and any future Megatron-Bridge model. Each
# model's README documents the exact invocation (HF id, pinned revision SHA, and
# FSx budget).
#
# Run this script INSIDE the training container (from 1.build-and-push.sh) on a
# node that has FSx Lustre mounted at /fsx.  It does NOT need torchrun — the
# AutoBridge import_ckpt() path is single-process.  If single-process import
# OOMs on host RAM, see the TODO below for the multi-GPU fallback.
#
# FSx budget (approximate; scales with model size):
#   - HF weights (block-FP8):                  ~0.5-1 TB
#   - Megatron-Core BF16 checkpoint:           ~1.3-2 TB
#   - Working / cache headroom:                ~1-2 TB
#   See the model README for the per-model figure (e.g. Kimi-K2 1.04T: 4-5 TB;
#   DeepSeek-V3 671B: 3-4 TB).
#
# Usage (from the model directory, e.g. kimi-k2/ or dsv3/):
#
#   # Kimi-K2 (1.04T; custom HF config -> trust_remote_code required):
#   HF_MODEL_ID=moonshotai/Kimi-K2-Base \
#   HF_REVISION=<commit-sha> \
#   FSX_ROOT=/fsx/kimi-k2 \
#   bash ../convert-checkpoint.sh
#
#   # DeepSeek-V3 (671B; dedicated deepseek_v3 bridge):
#   HF_MODEL_ID=deepseek-ai/DeepSeek-V3-Base \
#   HF_REVISION=<commit-sha> \
#   FSX_ROOT=/fsx/dsv3 \
#   bash ../convert-checkpoint.sh

set -euo pipefail

###############################################################################
# Required + user-configurable variables (override via environment before invoking)
###############################################################################

# HuggingFace model repo. REQUIRED — no default (this script is model-agnostic).
# Full-parameter SFT starts from the *Base* repo of the model.
HF_MODEL_ID="${HF_MODEL_ID:?set HF_MODEL_ID (e.g. moonshotai/Kimi-K2-Base or deepseek-ai/DeepSeek-V3-Base)}"

# Revision MUST be pinned to a commit SHA (enforced in Step 0 below); 'main' is a
# convention violation (non-reproducible). Find it at
#   https://huggingface.co/${HF_MODEL_ID}/commits/main
HF_REVISION="${HF_REVISION:-}"

# Canonical FSx root for this model's run (matches the model's conf + manifest,
# e.g. /fsx/kimi-k2 or /fsx/dsv3). REQUIRED.
FSX_ROOT="${FSX_ROOT:?set FSX_ROOT (e.g. /fsx/kimi-k2 or /fsx/dsv3)}"

# Where to cache the raw HF download (block-FP8)
FSX_HF_DIR="${FSX_HF_DIR:-${FSX_ROOT}/hf}"

# Where Megatron-Bridge writes the MCore distributed checkpoint (BF16)
FSX_MCORE_DIR="${FSX_MCORE_DIR:-${FSX_ROOT}/mcore}"

# HuggingFace access token (required if the repo is gated; empty if public).
HF_TOKEN="${HF_TOKEN:-}"

# trust_remote_code for AutoBridge / transformers config load. Default 1 (on):
# required for models whose HF config uses auto_map -> custom code (e.g. Kimi-K2's
# DeepseekV3Config). Harmless when the architecture is natively registered. Set
# TRUST_REMOTE_CODE=0 to force it off.
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

###############################################################################
# Step 0: Validate environment
###############################################################################

echo "=========================================================="
echo "Megatron-Bridge checkpoint conversion"
echo "  HF model:     ${HF_MODEL_ID} @ ${HF_REVISION}"
echo "  HF cache dir: ${FSX_HF_DIR}"
echo "  MCore dir:    ${FSX_MCORE_DIR}"
echo "  trust_remote_code: ${TRUST_REMOTE_CODE}"
echo "=========================================================="

# Confirm FSx is reachable. FSX_ROOT is a per-run subdir (created below if absent),
# so check its parent mount point rather than FSX_ROOT itself.
FSX_MOUNT="$(dirname "${FSX_ROOT}")"
if [[ ! -d "${FSX_MOUNT}" ]]; then
    echo "ERROR: FSx mount '${FSX_MOUNT}' does not exist. Is the volume mounted?" >&2
    exit 1
fi
mkdir -p "${FSX_ROOT}"

# Enforce a pinned revision: an empty or 'main' revision is a convention
# violation (non-reproducible). Require a commit SHA.
if [[ -z "${HF_REVISION}" || "${HF_REVISION}" == "main" ]]; then
    echo "ERROR: HF_REVISION is unset or 'main'. Pin it to a commit SHA for a" >&2
    echo "       reproducible conversion, e.g.:" >&2
    echo "         HF_REVISION=<commit-sha> bash ../convert-checkpoint.sh" >&2
    echo "       Find the SHA at https://huggingface.co/${HF_MODEL_ID}/commits/main" >&2
    exit 1
fi

# Confirm Megatron-Bridge is installed (installed into the NGC base image)
python - <<'PYCHECK'
try:
    from megatron.bridge import AutoBridge  # noqa: F401
except ImportError as e:
    raise SystemExit(f"megatron.bridge not importable — is the training container in use? {e}")
PYCHECK

###############################################################################
# Step 1: Download HuggingFace weights (block-FP8)
###############################################################################

mkdir -p "${FSX_HF_DIR}"

echo ""
echo "[1/2] Downloading HF weights -> ${FSX_HF_DIR}"

# HF_HUB_ENABLE_HF_TRANSFER speeds downloads (uses the hf_transfer C backend
# if installed; silently falls back to Python if not).
export HF_HUB_ENABLE_HF_TRANSFER=1

TOKEN_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    TOKEN_ARGS=(--token "${HF_TOKEN}")
fi

huggingface-cli download \
    "${HF_MODEL_ID}" \
    --revision "${HF_REVISION}" \
    --local-dir "${FSX_HF_DIR}" \
    "${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"}"

echo "Download complete."

###############################################################################
# Step 2: HF -> MCore checkpoint conversion via Megatron-Bridge
#
# AutoBridge.import_ckpt() is the single-process convenience wrapper shown in
# examples/conversion/convert_checkpoints.py for Megatron-Bridge v0.4.2
# (the version shipped in nvcr.io/nvidia/nemo:26.04.01).
#
# It calls AutoBridge.from_hf_pretrained(hf_model_id, torch_dtype=bfloat16)
# followed by provider.finalize() + save_megatron_model() in one shot.
# Block-FP8 weights are dequantized inline to BF16 during this step. AutoBridge
# resolves the HF architecture to a registered bridge — deepseek_v3_bridge.py for
# DeepSeek-V3 / Kimi-K2 (DeepseekV3ForCausalLM) — so the same call serves both.
#
# TODO(validate against image): the import_ckpt signature below is the v0.4.2
# shape; confirm against the built image. For Kimi-K2 the HF config carries a
# custom DeepseekV3Config via auto_map, so TRUST_REMOTE_CODE must be 1.
# CAVEAT: this is the AutoBridge *weight conversion* path (distinct from the
# weightless provider build to_megatron_provider(load_weights=False) used by the
# SFT conf). dsv3/benchmarks/bench_dsv3_pretrain.py notes the HF/AutoBridge import
# path was problematic on this image (the benchmark sidesteps it with random init +
# mock data); the Kimi-K2 conversion was exercised end-to-end. Validate import_ckpt
# end-to-end for your model before a production run.
#   AutoBridge.import_ckpt(
#       hf_model_id=<str>,
#       megatron_path=<str>,
#       torch_dtype=<torch.dtype>,   # torch.bfloat16 recommended for SFT
#       trust_remote_code=<bool>,
#   )
# Ref: https://github.com/NVIDIA-NeMo/Megatron-Bridge
#
# NOTE on parallelism: import_ckpt does NOT accept tp/pp/ep arguments — the
# MCore checkpoint it writes is parallelism-agnostic (TP=1, PP=1, EP=1).
# Megatron-Bridge reshards on the fly at training time via the training
# config's tensor_model_parallel_size / expert_model_parallel_size fields.
#
# TODO(validate against image): if this single-process import OOMs (host RAM is
# tight at BF16 + intermediate buffers for 0.6-1T params), fall back to the
# multi-GPU distributed conversion using torchrun:
#
#   torchrun --nproc_per_node=8 \
#     /path/to/hf_megatron_roundtrip_multi_gpu.py \
#     --hf-model-id "${FSX_HF_DIR}" \
#     --megatron-save-path "${FSX_MCORE_DIR}" \
#     --tp 8 --ep 8 --pp 1 \
#     --trust-remote-code
#
# The multi-GPU script is at:
#   https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/v0.4.2/examples/conversion/hf_megatron_roundtrip_multi_gpu.py
###############################################################################

mkdir -p "${FSX_MCORE_DIR}"

echo ""
echo "[2/2] Converting HF -> MCore (AutoBridge.import_ckpt) -> ${FSX_MCORE_DIR}"
echo "      This step dequantizes block-FP8 weights to BF16 inline."
echo "      Expected runtime: 30-90 min on a single node with fast FSx throughput."

FSX_HF_DIR="${FSX_HF_DIR}" FSX_MCORE_DIR="${FSX_MCORE_DIR}" \
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE}" python - <<'PYEOF'
import os
import torch
from megatron.bridge import AutoBridge

hf_dir = os.environ["FSX_HF_DIR"]
mcore_dir = os.environ["FSX_MCORE_DIR"]
trust_remote_code = os.environ.get("TRUST_REMOTE_CODE", "1") == "1"

# trust_remote_code is required for models whose HF config uses auto_map ->
# custom code (e.g. Kimi-K2's KimiK2ForCausalLM/DeepseekV3Config); harmless when
# the architecture is natively registered.
AutoBridge.import_ckpt(
    hf_model_id=hf_dir,
    megatron_path=mcore_dir,
    torch_dtype=torch.bfloat16,
    trust_remote_code=trust_remote_code,
)
print("import_ckpt complete.")
PYEOF

echo ""
echo "=========================================================="
echo "Checkpoint conversion finished."
echo "  MCore checkpoint: ${FSX_MCORE_DIR}"
echo ""
echo "Next step: run the single-node sanity gate (2.sanity-singlenode.sh), then"
echo "deploy the model's PyTorchJob (see the model's kubernetes/README.md). The"
echo "training job reads this checkpoint via the model conf's *_MCORE_CKPT env"
echo "(e.g. KIMI_K2_MCORE_CKPT / DSV3_MCORE_CKPT = ${FSX_MCORE_DIR})."
echo "=========================================================="
