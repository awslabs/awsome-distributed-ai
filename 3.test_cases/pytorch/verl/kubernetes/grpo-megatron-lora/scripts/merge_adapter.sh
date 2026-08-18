#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# Merge a verl v0.8.0 native PEFT adapter into a standalone HF model (Ray job).
#
# Since verl v0.8.0, a Megatron+LoRA checkpoint already contains
# a standard HF PEFT adapter at:
#   <ckpt>/global_step_<N>/actor/huggingface/adapter/
# so merging is just peft merge_and_unload() — no Megatron-Bridge replay merger.
#
# Submits scripts/merge_adapter.py as a Ray job onto a GPU node (device_map=auto
# shards the 235B base across the node's 8 B200s). Writes the merged HF model to
#   /fsx/data/verl/merged/<MODEL>/step_<N>/
#
# Usage:
#   ./scripts/merge_adapter.sh <STEP> [MODEL] [DATASET]
#
#   ./scripts/merge_adapter.sh 750
#   ./scripts/merge_adapter.sh 350 Qwen3-235B-A22B mixed-code-math-v2
#   DRY_RUN=1 ./scripts/merge_adapter.sh 750     # print the ray submit, don't run
#
# Env overrides:
#   MODEL, DATASET, BACKEND, BASE_MODEL_PATH, CKPT_ROOT, MERGED_ROOT,
#   DEVICE_MAP (default auto), DTYPE (default bfloat16)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${REPO_DIR}/env_vars" ]; then
    # shellcheck disable=SC1091
    source "${REPO_DIR}/env_vars"
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <STEP> [MODEL] [DATASET]" >&2
    echo "Example: $0 750" >&2
    exit 1
fi

STEP="$1"
case "${STEP}" in ''|*[!0-9]*) echo "error: STEP must be an integer, got '${STEP}'" >&2; exit 1 ;; esac

# Defaults reproduce the run in docs/results.md. Pass args or export MODEL/DATASET
# for your own: ./scripts/merge_adapter.sh <step> <model> <run-label>
MODEL="${2:-${MODEL:-Qwen3-235B-A22B}}"
DATASET="${3:-${DATASET:-mixed-code-math-v2}}"
BACKEND="${BACKEND:-megatron}"

FSX_HOME="${FSX_HOME:-/fsx/data/verl}"
CKPT_ROOT="${CKPT_ROOT:-${FSX_HOME}/ckpts/${DATASET}/${BACKEND}/${MODEL}}"
MERGED_ROOT="${MERGED_ROOT:-${FSX_HOME}/merged/${MODEL}}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${FSX_HOME}/models/${MODEL}}"

ADAPTER_DIR="${CKPT_ROOT}/global_step_${STEP}/actor/huggingface/adapter"
OUTPUT_DIR="${MERGED_ROOT}/step_${STEP}"

DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-bfloat16}"

# Provenance gate (see _assert_adapter_provenance in merge_adapter.py).
# DATASET defaults to `mixed-code-math-v2`, which holds a COMPLETE set of
# r=32/alpha=64 adapters (global_step_100 included). A different run at
# r=128/alpha=256 lives in its own tree. So `merge_adapter.sh 100` with default
# arguments merges the WRONG experiment's adapter, and the non-zero gate, the
# delta gate and check_merge_parity.py all pass -- none of them look at WHICH
# adapter it is.
#
# This is therefore fail-closed. A gate that is off unless you already know to turn it
# on protects nobody, and README points at check_merge_parity.py as the thing to run
# "before spending eval GPU-hours" -- so the identity check has to be on by default.
EXPECT_LORA_RANK="${EXPECT_LORA_RANK:-}"
EXPECT_LORA_ALPHA="${EXPECT_LORA_ALPHA:-}"
EXPECT_ARGS=()
if [ -n "${EXPECT_LORA_RANK}" ]; then
    EXPECT_ARGS+=( --expect-rank "${EXPECT_LORA_RANK}" )
fi
if [ -n "${EXPECT_LORA_ALPHA}" ]; then
    EXPECT_ARGS+=( --expect-alpha "${EXPECT_LORA_ALPHA}" )
fi
if [ ${#EXPECT_ARGS[@]} -eq 0 ] && [ "${ALLOW_UNVERIFIED_ADAPTER:-0}" != "1" ]; then
    echo "ERROR: EXPECT_LORA_RANK unset -- nothing verifies WHICH adapter is merged." >&2
    echo "       DATASET currently resolves to '${DATASET}'." >&2
    echo "       Every other gate (non-zero, delta, check_merge_parity.py) passes on an" >&2
    echo "       adapter from the wrong experiment, so merging without this is untraceable." >&2
    echo "" >&2
    echo "       Set the rank the run was trained at:" >&2
    echo "         EXPECT_LORA_RANK=32 EXPECT_LORA_ALPHA=64 $0 $*" >&2
    echo "       Or, to merge without an identity check:" >&2
    echo "         ALLOW_UNVERIFIED_ADAPTER=1 $0 $*" >&2
    exit 1
fi

# Single verl pin — matches scripts/runtime_env.yaml (v0.8.0). Keep in sync.
VERL_PIN="${VERL_PIN:-v0.8.0}"

# Default assumes a local port-forward (see README). For an ALB-hosted dashboard,
# set RAY_ADDRESS (or RAY_DASHBOARD_HOSTNAME) in env_vars; RAY_ADDRESS wins below.
RAY_DASHBOARD_HOSTNAME="${RAY_DASHBOARD_HOSTNAME:-localhost:8265}"
# Accept a bare host (assume https, ALB case) or a full URL (port-forward case).
case "${RAY_DASHBOARD_HOSTNAME}" in
    *://*) _ray_default="${RAY_DASHBOARD_HOSTNAME}" ;;
    localhost:*|127.0.0.1:*) _ray_default="http://${RAY_DASHBOARD_HOSTNAME}" ;;
    *) _ray_default="https://${RAY_DASHBOARD_HOSTNAME}" ;;
esac
RAY_ADDRESS="${RAY_ADDRESS:-${_ray_default}}"

echo "=============================================="
echo "Merge Adapter -> HF Model (Ray job)"
echo "=============================================="
echo "Step:          ${STEP}"
echo "Model:         ${MODEL}"
echo "Dataset:       ${DATASET}"
echo "Adapter dir:   ${ADAPTER_DIR}"
echo "Base model:    ${BASE_MODEL_PATH}"
echo "Output:        ${OUTPUT_DIR}"
echo "device_map:    ${DEVICE_MAP}"
echo "dtype:         ${DTYPE}"
echo "verl pin:      ${VERL_PIN}"
echo "Ray address:   ${RAY_ADDRESS}"
echo "=============================================="

HEADERS_ARGS=()
if [ -n "${RAY_HEADERS:-}" ]; then
    HEADERS_ARGS=(--headers "${RAY_HEADERS}")
fi

# peft/transformers/safetensors ship in the training image, but pin verl[mcore]
# for a consistent transformers/peft stack identical to training rollouts.
RUNTIME_ENV=$(cat <<RTEOF
{
  "working_dir": "${REPO_DIR}",
  "excludes": ["outputs/", "profiling/", ".git/", "*.pyc", "__pycache__/"],
  "pip": {
    "packages": [
      # peft performs the actual merge_and_unload(), so an unbounded major bump changes
      # merge behaviour with no signal -- "pip_check": false below would not catch it.
      "peft>=0.11,<0.20",
      "safetensors>=0.4,<1",
      "verl[mcore] @ git+https://github.com/volcengine/verl.git@${VERL_PIN}"
    ],
    "pip_check": false
  }
}
RTEOF
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY_RUN] would submit:"
    echo "  ray job submit --address ${RAY_ADDRESS} --entrypoint-num-gpus 8 --entrypoint-num-cpus 16 -- \\"
    echo "    python scripts/merge_adapter.py --base-model ${BASE_MODEL_PATH} \\"
    echo "      --adapter-dir ${ADAPTER_DIR} --output-dir ${OUTPUT_DIR} \\"
    echo "      --dtype ${DTYPE} --device-map ${DEVICE_MAP} --overwrite ${EXPECT_ARGS[*]}"
    exit 0
fi

# --entrypoint-num-gpus 8: reserve the whole node so device_map=auto can shard
# the 235B base across all 8 B200s. (The merge is a one-shot, not a Ray actor.)
ray job submit \
    --address "${RAY_ADDRESS}" \
    ${HEADERS_ARGS[@]+"${HEADERS_ARGS[@]}"} \
    --runtime-env-json "${RUNTIME_ENV}" \
    --entrypoint-num-gpus 8 \
    --entrypoint-num-cpus 16 \
    --no-wait \
    -- python scripts/merge_adapter.py \
        --base-model "${BASE_MODEL_PATH}" \
        --adapter-dir "${ADAPTER_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --dtype "${DTYPE}" \
        --device-map "${DEVICE_MAP}" \
        --overwrite \
        ${EXPECT_ARGS[@]+"${EXPECT_ARGS[@]}"}

echo ""
echo "=============================================="
echo "Merge job submitted."
echo "Monitor: ${RAY_ADDRESS}/#/jobs"
echo "Merged model when done: ${OUTPUT_DIR}"
echo "Then run the parity gate:"
echo "  python scripts/check_merge_parity.py --base-model ${BASE_MODEL_PATH} --merged-model ${OUTPUT_DIR}"
echo "=============================================="
