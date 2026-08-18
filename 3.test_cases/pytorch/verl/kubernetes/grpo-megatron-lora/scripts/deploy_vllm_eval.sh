#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# Deploy or Tear Down the vLLM Evaluation Server
#
# Wraps `envsubst < kubernetes/vllm-eval.yaml | kubectl apply -f -` with
# validation of required env vars and a rollout-status wait.  Used by the
# evaluation pipeline (see docs/eval-pipeline.md).
#
# Usage:
#   # Deploy for a specific checkpoint
#   export VLLM_MODEL_PATH=/fsx/data/verl/merged/Qwen3-235B-A22B/step_200
#   export VLLM_SERVED_NAME=qwen3-235b-step200
#   ./scripts/deploy_vllm_eval.sh
#
#   # Deploy the base model for comparison
#   export VLLM_MODEL_PATH=/fsx/data/verl/models/Qwen3-235B-A22B
#   export VLLM_SERVED_NAME=qwen3-235b-base
#   ./scripts/deploy_vllm_eval.sh
#
#   # Tear down
#   ./scripts/deploy_vllm_eval.sh --delete
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${REPO_DIR}/kubernetes/vllm-eval.yaml"

if [ -f "${REPO_DIR}/env_vars" ]; then
    # shellcheck disable=SC1091
    source "${REPO_DIR}/env_vars"
fi

# Defaults (override via env_vars or export in shell)
export KUBE_NAMESPACE="${KUBE_NAMESPACE:-default}"
export REGISTRY="${REGISTRY:?REGISTRY must be set (ECR registry URL with trailing slash)}"
export IMAGE="${IMAGE:-verl-grpo-lora}"
# Never default to `latest`: an eval must be reproducible against the exact image
# that trained the checkpoint. Keep this in step with build-push.sh.
export TAG="${TAG:-v0.8.0-vllm020.dev2}"
# Node selection lives in the manifest (nodeAffinity matches both
# `p6-b200.48xlarge` Karpenter-launched nodes and `ml.p6-b200.48xlarge`
# HyperPod-managed nodes) — no shell-side knob needed.
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-8}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12288}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
# Runtime-LoRA serving (optional). Set VLLM_ENABLE_LORA=true and
# VLLM_LORA_MODULES="name=/path/to/adapter" to serve base + adapter without a
# pre-merge. The served model name for requests is then the LoRA module name.
export VLLM_ENABLE_LORA="${VLLM_ENABLE_LORA:-false}"
export VLLM_LORA_MODULES="${VLLM_LORA_MODULES:-}"
export VLLM_MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-32}"
export VLLM_MAX_LORAS="${VLLM_MAX_LORAS:-1}"
# CUDA graphs are ON by default now. --enforce-eager measured 4.55 tok/s/stream
# (~220ms/token) on 235B MoE TP=8, latency-bound and flat across concurrency,
# projecting ~40h per model at max_gen_toks=24576. Set true only to restore it.
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-false}"
# REQUIRED for large adapters: a 105 GB adapter load far exceeds vLLM's 300 s
# default and the first request returns HTTP 500 "RPC call to sample_tokens
# timed out". See docs/results.md.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"

# Tear-down mode
if [ "${1:-}" = "--delete" ] || [ "${1:-}" = "-d" ]; then
    echo "Tearing down vllm-eval Deployment + Service in namespace ${KUBE_NAMESPACE}..."
    kubectl delete -n "${KUBE_NAMESPACE}" deploy/vllm-eval --ignore-not-found
    kubectl delete -n "${KUBE_NAMESPACE}" svc/vllm-eval --ignore-not-found
    echo "Done."
    exit 0
fi

# Required for deploy
: "${VLLM_MODEL_PATH:?Set VLLM_MODEL_PATH to the merged HF model dir on FSx}"
: "${VLLM_SERVED_NAME:?Set VLLM_SERVED_NAME (e.g. qwen3-235b-step200)}"
export VLLM_MODEL_PATH VLLM_SERVED_NAME

echo "=============================================="
echo "Deploy vLLM Eval Server"
echo "=============================================="
echo "Namespace:          ${KUBE_NAMESPACE}"
echo "Image:              ${REGISTRY}${IMAGE}:${TAG}"
echo "Model path:         ${VLLM_MODEL_PATH}"
echo "Served name:        ${VLLM_SERVED_NAME}"
echo "Tensor parallel:    ${VLLM_TP_SIZE}"
echo "Max model len:      ${VLLM_MAX_MODEL_LEN}"
echo "Enforce eager:      ${VLLM_ENFORCE_EAGER} (false = CUDA graphs ON)"
echo "Max num seqs:       ${VLLM_MAX_NUM_SEQS}"
echo "GPU mem util:       ${VLLM_GPU_MEM_UTIL}"
echo "=============================================="

# Render + apply. Allowlist envsubst so the container-runtime shell vars in the
# vLLM command block (${LORA_ARGS[@]}, the `for m in ...` loop var ${m}, and the
# ${VAR:-default} fallbacks) survive; only these template vars are substituted.
SUBST_VARS='${REGISTRY} ${IMAGE} ${TAG} ${KUBE_NAMESPACE} ${VLLM_MODEL_PATH} ${VLLM_SERVED_NAME} ${VLLM_TP_SIZE} ${VLLM_MAX_MODEL_LEN} ${VLLM_MAX_NUM_SEQS} ${VLLM_GPU_MEM_UTIL} ${VLLM_ENABLE_LORA} ${VLLM_LORA_MODULES} ${VLLM_MAX_LORA_RANK} ${VLLM_MAX_LORAS} ${VLLM_ENFORCE_EAGER} ${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS}'
envsubst "${SUBST_VARS}" < "${MANIFEST}" | kubectl apply -n "${KUBE_NAMESPACE}" -f -

echo ""
echo "Waiting for rollout (may take 10-15 min for 235B model loading)..."
kubectl rollout status -n "${KUBE_NAMESPACE}" deploy/vllm-eval --timeout=20m

echo ""
echo "=============================================="
echo "vLLM eval server ready."
echo "Endpoint (in-cluster):"
echo "  http://vllm-eval.${KUBE_NAMESPACE}.svc.cluster.local:8000"
echo ""
echo "Test:"
echo "  kubectl exec fsx-utils -- wget -qO- http://vllm-eval.${KUBE_NAMESPACE}.svc.cluster.local:8000/v1/models"
echo ""
echo "Logs:"
echo "  kubectl logs -f deploy/vllm-eval -n ${KUBE_NAMESPACE}"
echo "=============================================="
