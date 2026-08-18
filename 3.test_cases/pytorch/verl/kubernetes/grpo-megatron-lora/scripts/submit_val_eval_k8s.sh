#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# Submit the Validation-Split reward eval as a K8s Job (not Ray).
#
# Use when the Ray worker group is scaled to 0 (single GPU node dedicated to the
# vllm-eval server). eval_val_split.py needs only CPU + HTTP to vllm-eval and
# sandbox-fusion, so it runs fine as a plain K8s Job co-located on the GPU node.
#
# Stages scripts/eval_val_split.py + scripts/custom_reward_fn.py to FSx (so the
# Job can run them with the sibling import intact), then renders valeval-job.yaml.
#
# Usage:
#   ./scripts/submit_val_eval_k8s.sh <run-name>
#   VAL_LIMIT=20 ./scripts/submit_val_eval_k8s.sh qwen3-235b-step750   # smoke
#
# Env overrides:
#   VLLM_SERVED_NAME (default = run-name; use the LoRA module name for a step),
#   VAL_PARQUET, VAL_N_SAMPLES (default 4), VAL_CONCURRENCY (default 32),
#   VALEVAL_IMAGE (default python:3.11-slim), VERL_PIN (default v0.8.0),
#   FSX_UTILS_POD (default fsx-utils).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${REPO_DIR}/kubernetes/valeval-job.yaml"

if [ -f "${REPO_DIR}/env_vars" ]; then
    # shellcheck disable=SC1091
    source "${REPO_DIR}/env_vars"
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run-name>" >&2
    exit 1
fi

EVAL_RUN_NAME="$1"
EVAL_RUN_SLUG="$(echo "${EVAL_RUN_NAME}" | tr '[:upper:]_' '[:lower:]-' | tr -cd 'a-z0-9-' | cut -c1-50)"

export KUBE_NAMESPACE="${KUBE_NAMESPACE:-default}"
export EVAL_RUN_NAME EVAL_RUN_SLUG
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-${EVAL_RUN_NAME}}"
export VAL_PARQUET="${VAL_PARQUET:-/fsx/data/verl/data/mixed-code-math/val.parquet}"
export VAL_N_SAMPLES="${VAL_N_SAMPLES:-4}"
export VAL_CONCURRENCY="${VAL_CONCURRENCY:-32}"
# Generation cap for the held-out reward signal. MUST match the training rollout
# budget (training.max_response_length = 16384).
#
# This was previously 4096 with the rationale that "4k is plenty" — that is
# empirically wrong and it silently corrupted every external val-split number.
# Training rollouts on this data average 8,300-9,600 response tokens, so a 4k cap
# truncated the MAJORITY of generations, and a truncated code/math answer scores 0.
# Measured consequence: external val-split reported apps mean_score=0.102 while
# in-training validation (which generates at the full 16k) reported apps
# acc=0.644 on the SAME val parquet — a 6x disagreement that is entirely an
# artifact of this cap, not a property of the model.
#
# The original 4k was chosen to dodge HTTP timeouts on long competitive-
# programming completions. That is an operational problem, not a reason to
# corrupt the measurement: eval_val_split.py already uses a 1800s per-request
# timeout, so if you see timeouts or throughput collapse, REDUCE VAL_CONCURRENCY
# (32 -> 8/16) rather than lowering this cap.
# MEASURED 2026-07-31: at a 16384 cap, 22% of rollouts still truncate
# (response_length/clip_ratio = 0.218 over 422 steps, and response_length/max was
# pinned at 16384 in EVERY step -- the distribution was fully right-censored).
# Raising the cap to 24576 dropped clip_ratio to 0.035 and lifted mean response
# length 9132 -> 12022, so 16384 was still suppressing measured scores. 24576
# leaves ~3.5% truncating; raise further if you need that last few percent.
export VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-24576}"
# Stratified subset: with VAL_LIMIT, sample evenly across data_source groups.
export VAL_STRATIFY="${VAL_STRATIFY:-false}"
export VAL_LIMIT="${VAL_LIMIT:-0}"
export VALEVAL_IMAGE="${VALEVAL_IMAGE:-python:3.11-slim}"
export VERL_PIN="${VERL_PIN:-v0.8.0}"

FSX_UTILS_POD="${FSX_UTILS_POD:-fsx-utils}"
STAGE_DIR="/fsx/data/verl/eval_scripts"
export VALEVAL_SCRIPT="${STAGE_DIR}/eval_val_split.py"

echo "=============================================="
echo "Submit Val-Split eval (K8s Job)"
echo "=============================================="
echo "Run name:      ${EVAL_RUN_NAME}"
echo "Job name:      valeval-${EVAL_RUN_SLUG}"
echo "Served name:   ${VLLM_SERVED_NAME}"
echo "Val parquet:   ${VAL_PARQUET}"
echo "n_samples:     ${VAL_N_SAMPLES}   concurrency: ${VAL_CONCURRENCY}   limit: ${VAL_LIMIT}"
echo "Script (FSx):  ${VALEVAL_SCRIPT}"
echo "Image:         ${VALEVAL_IMAGE}   verl: ${VERL_PIN}"
echo "=============================================="

# Stage the two scripts to FSx so the Job can run them (sibling import intact).
echo "[stage] copying eval_val_split.py + custom_reward_fn.py to ${STAGE_DIR}"
kubectl exec "${FSX_UTILS_POD}" -n "${KUBE_NAMESPACE}" -- mkdir -p "${STAGE_DIR}"
kubectl cp "${REPO_DIR}/scripts/eval_val_split.py" "${KUBE_NAMESPACE}/${FSX_UTILS_POD}:${STAGE_DIR}/eval_val_split.py"
kubectl cp "${REPO_DIR}/scripts/custom_reward_fn.py" "${KUBE_NAMESPACE}/${FSX_UTILS_POD}:${STAGE_DIR}/custom_reward_fn.py"

kubectl delete job "valeval-${EVAL_RUN_SLUG}" -n "${KUBE_NAMESPACE}" --ignore-not-found >/dev/null 2>&1 || true

SUBST_VARS='${KUBE_NAMESPACE} ${EVAL_RUN_NAME} ${EVAL_RUN_SLUG} ${VLLM_SERVED_NAME} ${VAL_PARQUET} ${VAL_N_SAMPLES} ${VAL_CONCURRENCY} ${VAL_MAX_TOKENS} ${VAL_STRATIFY} ${VAL_LIMIT} ${VALEVAL_IMAGE} ${VERL_PIN} ${VALEVAL_SCRIPT}'
envsubst "${SUBST_VARS}" < "${MANIFEST}" | kubectl apply -n "${KUBE_NAMESPACE}" -f -

echo ""
echo "Submitted. Follow logs:"
echo "  kubectl logs -f job/valeval-${EVAL_RUN_SLUG} -n ${KUBE_NAMESPACE}"
