#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# Submit Validation-Split Evaluation as a Ray Job
#
# Runs scripts/eval_val_split.py on the Ray cluster (so it can reach the
# in-cluster vllm-eval + sandbox-fusion services via ClusterIP DNS).  Writes
# results to /fsx/data/verl/eval_results/<run-name>/val_split.json.
#
# Prerequisites:
#   - vllm-eval Deployment is running (scripts/deploy_vllm_eval.sh)
#   - sandbox-fusion Deployment is running (standard training setup)
#   - Mixed dataset is prepared: /fsx/data/verl/data/mixed-code-math/val.parquet
#
# Usage:
#   ./scripts/submit_val_eval.sh qwen3-235b-step200
#   ./scripts/submit_val_eval.sh qwen3-235b-base
#
# Optional env overrides:
#   VAL_PARQUET   — override the val parquet path
#   N_SAMPLES     — responses per prompt (default 4, matches training n=4)
#   CONCURRENCY   — concurrent HTTP requests (default 32)
#   VAL_LIMIT     — quick-test subset size (0 = all)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${REPO_DIR}/env_vars" ]; then
    # shellcheck disable=SC1091
    source "${REPO_DIR}/env_vars"
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run-name>"
    echo ""
    echo "Example: $0 qwen3-235b-step200"
    exit 1
fi

RUN_NAME="$1"

KUBE_NAMESPACE="${KUBE_NAMESPACE:-default}"
VAL_PARQUET="${VAL_PARQUET:-/fsx/data/verl/data/mixed-code-math/val.parquet}"
VLLM_URL="${VLLM_URL:-http://vllm-eval.${KUBE_NAMESPACE}.svc.cluster.local:8000/v1}"
SANDBOX_URL="${SANDBOX_URL:-http://sandbox-fusion.${KUBE_NAMESPACE}.svc.cluster.local:8080/run_code}"
SERVED_NAME="${VLLM_SERVED_NAME:-${RUN_NAME}}"
OUTPUT="${OUTPUT:-/fsx/data/verl/eval_results/${RUN_NAME}/val_split.json}"
N_SAMPLES="${N_SAMPLES:-4}"
CONCURRENCY="${CONCURRENCY:-32}"
VAL_LIMIT="${VAL_LIMIT:-0}"

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
echo "Submit Val-Split Evaluation Job"
echo "=============================================="
echo "Run name:      ${RUN_NAME}"
echo "Val parquet:   ${VAL_PARQUET}"
echo "vLLM URL:      ${VLLM_URL}"
echo "Served name:   ${SERVED_NAME}"
echo "Sandbox URL:   ${SANDBOX_URL}"
echo "Output:        ${OUTPUT}"
echo "n_samples:     ${N_SAMPLES}"
echo "concurrency:   ${CONCURRENCY}"
echo "limit:         ${VAL_LIMIT:-0}"
echo "Ray address:   ${RAY_ADDRESS}"
echo "=============================================="

HEADERS_ARGS=()
if [ -n "${RAY_HEADERS:-}" ]; then
    HEADERS_ARGS=(--headers "${RAY_HEADERS}")
fi

# Ray runtime env — ship the repo, install aiohttp + pandas + pyarrow + verl
# (verl is needed because custom_reward_fn imports default_compute_score from it).
RUNTIME_ENV=$(cat <<RTEOF
{
  "working_dir": "${REPO_DIR}",
  "excludes": ["outputs/", "profiling/", ".git/", "*.pyc", "__pycache__/"],
  "pip": {
    "packages": [
      "aiohttp>=3.9,<4",
      "tqdm>=4.66,<5",
      "pandas>=2.0,<3",
      "pyarrow>=14,<25",
      "verl[mcore] @ git+https://github.com/volcengine/verl.git@v0.8.0"
    ],
    "pip_check": false
  }
}
RTEOF
)

EXTRA_ARGS=()
if [ "${VAL_LIMIT}" -gt 0 ]; then
    EXTRA_ARGS+=(--limit "${VAL_LIMIT}")
fi

# Generation budget. Must match training's max_response_length (conf/config.yaml) or the
# two val-split paths are not comparable with each other, nor with in-training validation.
# Stated here rather than inherited from the argparse default so the value is visible in
# both submitters -- submit_val_eval_k8s.sh sets the same figure.
VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-24576}"

ray job submit \
    --address "${RAY_ADDRESS}" \
    ${HEADERS_ARGS[@]+"${HEADERS_ARGS[@]}"} \
    --runtime-env-json "${RUNTIME_ENV}" \
    --entrypoint-num-cpus 4 \
    --no-wait \
    -- python scripts/eval_val_split.py \
        --val-parquet "${VAL_PARQUET}" \
        --vllm-url "${VLLM_URL}" \
        --served-name "${SERVED_NAME}" \
        --sandbox-url "${SANDBOX_URL}" \
        --output "${OUTPUT}" \
        --n-samples "${N_SAMPLES}" \
        --concurrency "${CONCURRENCY}" \
        --max-tokens "${VAL_MAX_TOKENS}" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo ""
echo "=============================================="
echo "Val-split eval job submitted."
echo "Monitor: ${RAY_ADDRESS}/#/jobs"
echo "Results when done: ${OUTPUT}"
echo "=============================================="
