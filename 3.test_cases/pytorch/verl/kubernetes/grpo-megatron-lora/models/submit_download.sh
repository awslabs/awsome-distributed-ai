#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# Submit Model Download as a Ray Job
#
# Uses Ray runtime_env to ship the download script to the cluster.
# The base image (verl-grpo-lora) already has huggingface_hub, safetensors,
# and torch — no extra pip installs needed.
#
# Usage:
#   # Download Qwen3-8B (pipeline validation, ~16GB)
#   ./models/submit_download.sh Qwen/Qwen3-8B
#
#   # Download target model (~159GB)
#   ./models/submit_download.sh Qwen/Qwen3-Coder-Next
#
#   # Use an alias
#   ./models/submit_download.sh qwen3-8b
#
#   # Custom output directory
#   ./models/submit_download.sh Qwen/Qwen3-8B --output-dir /fsx/data/verl/models
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load environment variables
if [ -f "${REPO_DIR}/env_vars" ]; then
    # shellcheck disable=SC1091
    source "${REPO_DIR}/env_vars"
fi

# Ray address
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

# Model argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model-id-or-alias> [--output-dir /path] [--skip-verify]"
    echo ""
    echo "Aliases: qwen3-8b, qwen3-coder-next, qwen2.5-coder-7b, qwen2.5-72b, qwen3-235b, qwen3-30b-a3b"
    echo ""
    echo "Examples:"
    echo "  $0 Qwen/Qwen3-8B"
    echo "  $0 qwen3-8b"
    echo "  $0 Qwen/Qwen3-Coder-Next"
    exit 1
fi

MODEL="$1"
shift
# Array, not "$*". Flattening the remaining arguments into one string and expanding it
# unquoted at the submit line word-split any path containing a space and glob-expanded
# any wildcard on the submit host. The sibling submitters already use the array form.
EXTRA_ARGS=("$@")

# Build env vars for runtime_env
ENV_VARS_JSON="{}"
if [ -n "${HF_TOKEN:-}" ]; then
    ENV_VARS_JSON="{\"HF_TOKEN\": \"${HF_TOKEN}\"}"
fi

# Runtime env: ship the models/ directory, set env vars
# The working_dir ships our code; the base image provides all Python deps
RUNTIME_ENV=$(cat <<RTEOF
{
  "working_dir": "${REPO_DIR}",
  "excludes": ["data/__pycache__/", ".git/", "*.pyc"],
  "env_vars": ${ENV_VARS_JSON}
}
RTEOF
)

echo "=============================================="
echo "Submit Model Download Job"
echo "=============================================="
echo "Ray address: ${RAY_ADDRESS}"
echo "Model:       ${MODEL}"
echo "Extra args:  ${EXTRA_ARGS[*]:-none}"
echo "=============================================="

# Build Ray CLI auth headers if available (OIDC-protected dashboard). This script was
# the only Ray submitter that omitted them, so model download was the one workflow that
# failed with an auth error behind an authenticating proxy while the others succeeded.
HEADERS_ARGS=()
if [ -n "${RAY_HEADERS:-}" ]; then
    HEADERS_ARGS=(--headers "${RAY_HEADERS}")
fi

# Submit — request only CPUs (no GPU needed for download)
# entrypoint-num-cpus=4 ensures it lands on a node with spare CPU
ray job submit \
    --address "${RAY_ADDRESS}" \
    ${HEADERS_ARGS[@]+"${HEADERS_ARGS[@]}"} \
    --runtime-env-json "${RUNTIME_ENV}" \
    --entrypoint-num-cpus 4 \
    --no-wait \
    -- python models/download_model.py --model "${MODEL}" \
       ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo ""
echo "=============================================="
echo "Job submitted! Monitor at: ${RAY_ADDRESS}/#/jobs"
echo "=============================================="
