#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# Submit the lm-evaluation-harness eval Job
#
# Runs HumanEval + MBPP (pass@1, pass@10) against the warm `vllm-eval` service
# via lm-eval-harness `local-completions`. Driver is CPU-only and reuses the
# already-deployed vLLM server (deploy_vllm_eval.sh) — no model reload, no GPU.
#
# Prerequisites:
#   - vllm-eval Deployment running + ready (scripts/deploy_vllm_eval.sh)
#
# Usage:
#   ./scripts/submit_lmeval.sh <run-name>
#   ./scripts/submit_lmeval.sh qwen3-235b-step750
#   LMEVAL_LIMIT=5 ./scripts/submit_lmeval.sh qwen3-235b-step750   # smoke test
#
# pass@k strategy (lm-eval computes pass@1 and pass@10 for humaneval/mbpp when
# the task's metric list includes them; we sample at temperature>0 with the
# harness's built-in n). Default tasks: humaneval,mbpp.
#
# Env overrides:
#   LMEVAL_TASKS        default "humaneval,mbpp"
#   LMEVAL_LIMIT        subset size for smoke (0 = full)
#   LMEVAL_CONCURRENCY  concurrent requests to vLLM (default 24 — must not exceed
#                       server capacity; see the sizing note below)
#   LMEVAL_TEMPERATURE  sampling temp (default 0.2 — pass@1 paper-comparable)
#   LMEVAL_TOP_P        default 0.95
#   LMEVAL_IMAGE        driver image (default vllm/vllm-openai:v0.20.2)
#   LMEVAL_VERSION      lm-eval pip version (default 0.4.9)
#   LMEVAL_INSTANCE_TYPE  CPU node instance-type (default m5.8xlarge)
#   VLLM_SERVED_NAME    served-model-name on the vLLM server (default = run-name)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${REPO_DIR}/kubernetes/lmeval-job.yaml"

if [ -f "${REPO_DIR}/env_vars" ]; then
    # shellcheck disable=SC1091
    source "${REPO_DIR}/env_vars"
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run-name>" >&2
    echo "Example: $0 qwen3-235b-step750" >&2
    exit 1
fi

EVAL_RUN_NAME="$1"
# K8s object names: lowercase alnum + '-'. Slugify the run name.
EVAL_RUN_SLUG="$(echo "${EVAL_RUN_NAME}" | tr '[:upper:]_' '[:lower:]-' | tr -cd 'a-z0-9-' | cut -c1-50)"

export KUBE_NAMESPACE="${KUBE_NAMESPACE:-default}"
export EVAL_RUN_NAME EVAL_RUN_SLUG
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-${EVAL_RUN_NAME}}"
export LMEVAL_TASKS="${LMEVAL_TASKS:-humaneval,mbpp}"
export LMEVAL_LIMIT="${LMEVAL_LIMIT:-0}"
# Concurrent in-flight requests. MUST NOT EXCEED WHAT THE SERVER CAN RUN.
#
# This is a correctness knob, not a throughput knob. lm-eval's HTTP clock starts
# when a request is SENT, not when generation begins, so anything the server
# queues is burning its own timeout. Over-subscribe and requests at the back of
# the queue expire, tenacity retries them at full generation cost, and when
# max_retries is exhausted lm-eval raises asyncio.TimeoutError and kills the
# whole run -- losing every completed sample.
#
# Measured on Qwen3-235B-A22B, TP=8, max_model_len=32768, 2026-08-05. vLLM
# reported `GPU KV cache size: 318,640 tokens` = ~9.7x concurrency at full
# length, so it sustains ~18-24 sequences:
#
#   num_concurrent=128 -> Running 18, Waiting 110, KV 97-100%, 530 tok/s
#                         1202 TimeoutError/Retrying events, then died at
#                         1069/1640 after ~3h with asyncio.TimeoutError
#   num_concurrent=24  -> Running 24, Waiting   0,            755 tok/s
#
# Note throughput went UP 42% when concurrency came DOWN. Over-subscribing a
# KV-bound server does not merely add latency: with KV pinned at ~100% vLLM
# thrashes on preemption/recompute, so the excess requests cost real compute.
#
# Sizing rule: read `GPU KV cache size` from the vLLM startup log, divide by
# max_model_len for the concurrency ceiling, and set this at or just below it.
export LMEVAL_CONCURRENCY="${LMEVAL_CONCURRENCY:-24}"
# Per-REQUEST HTTP timeout (s) for lm-eval->vLLM, not a job timeout. Each expiry
# burns a retry at full generation cost, so this is the last line of defence --
# the real fix for timeouts is LMEVAL_CONCURRENCY above.
export LMEVAL_TIMEOUT="${LMEVAL_TIMEOUT:-7200}"
export LMEVAL_TEMPERATURE="${LMEVAL_TEMPERATURE:-0.2}"
export LMEVAL_TOP_P="${LMEVAL_TOP_P:-0.95}"
export LMEVAL_IMAGE="${LMEVAL_IMAGE:-python:3.11-slim}"
export LMEVAL_VERSION="${LMEVAL_VERSION:-0.4.9}"
export LMEVAL_INSTANCE_TYPE="${LMEVAL_INSTANCE_TYPE:-m5.xlarge}"
# lm-eval builds a tokenizer from the model name; point it at the base model dir
# on FSx so it doesn't try to resolve the served LoRA name as an HF repo id.
# The default is the model used in docs/results.md -- OVERRIDE IT for your model:
#   export LMEVAL_TOKENIZER=/fsx/data/verl/models/<your-model>
export LMEVAL_TOKENIZER="${LMEVAL_TOKENIZER:-/fsx/data/verl/models/Qwen3-235B-A22B}"
# lm-evaluation-harness context window. MUST be >= max_gen_toks in the task YAMLs
# PLUS the longest prompt, and should match the vLLM server's --max-model-len.
#
# This is a silent-failure trap. TemplateAPI (local-completions) defaults
# max_length=2048 and truncates the context to `max_length - max_gen_toks`. With
# the corrected max_gen_toks=24576 that is 2048-24576 = NEGATIVE, which drives
# the request's max_tokens to <= 0 and every call dies with
#   HTTP 400 "max_tokens must be at least 1, got 0"
# It looks exactly like the chat-template pitfall (chat-templated prompts rejected by
# /v1/completions) but is unrelated: chat prompts are accepted fine, and
# reverting LMEVAL_APPLY_CHAT_TEMPLATE would only restore the broken
# completion-mode config that produced HumanEval pass@1 = 0.2634.
export LMEVAL_MAX_LENGTH="${LMEVAL_MAX_LENGTH:-32768}"
# Chat template. MUST stay consistent with the `until` / max_gen_toks settings in
# kubernetes/lmeval-tasks/*.yaml — the two are coupled, and mixing them is how
# HumanEval pass@1 came out at 0.2634 for a frontier 235B model:
#
#   completion mode (false): base-style prompt, until=["\ndef","\nclass",...],
#     small max_gen_toks. Correct for BASE models.
#   chat mode (true):        chat-templated prompt, stop only on <|im_end|>,
#     max_gen_toks sized to the model's real reasoning length, and code recovered
#     by the utils.build_predictions filter. Correct for INSTRUCT/REASONING models
#     like Qwen3-235B-A22B, and matches how training rollouts are generated.
#
# Default is now `true` (chat mode) because the task YAMLs are configured for it,
# the build_predictions extraction filter only makes sense for prose responses,
# and AGENTS.md prescribes `local-completions` + --apply_chat_template.
#
# CAUTION: chat-templated prompts sent to /v1/completions have
# previously returned HTTP 400. ALWAYS smoke-test first before spending eval
# GPU-hours:  LMEVAL_LIMIT=2 ./scripts/submit_lmeval.sh <run-name>
# If it 400s, set LMEVAL_APPLY_CHAT_TEMPLATE=false AND revert the task YAMLs to
# completion-mode `until` stops — do not mix the two.
export LMEVAL_APPLY_CHAT_TEMPLATE="${LMEVAL_APPLY_CHAT_TEMPLATE:-true}"
# gen_kwargs + metadata. Default = greedy pass@1. For pass@10, set e.g.
#   LMEVAL_GEN_KWARGS="do_sample=True,temperature=0.8,top_p=0.95"
#   LMEVAL_METADATA='{"num_samples":10}'   (task computes pass@1 AND pass@10)
export LMEVAL_GEN_KWARGS="${LMEVAL_GEN_KWARGS:-temperature=${LMEVAL_TEMPERATURE},top_p=${LMEVAL_TOP_P}}"
export LMEVAL_METADATA="${LMEVAL_METADATA:-}"
# Base64 the metadata JSON so its double-quotes survive envsubst + shell intact.
if [ -n "${LMEVAL_METADATA}" ]; then
    LMEVAL_METADATA_B64="$(printf '%s' "${LMEVAL_METADATA}" | base64 | tr -d '\n')"
else
    LMEVAL_METADATA_B64=""
fi
export LMEVAL_METADATA_B64

# Custom task YAMLs (humaneval_p10, mbpp_p10, utils.py) -> ConfigMap mounted at
# /tasks and registered via --include_path. Recreated each submit so edits land.
TASKS_DIR="${REPO_DIR}/kubernetes/lmeval-tasks"
export LMEVAL_TASKS_CONFIGMAP="lmeval-tasks-${EVAL_RUN_SLUG}"
if [ -d "${TASKS_DIR}" ] && [ -n "$(ls -A "${TASKS_DIR}" 2>/dev/null)" ]; then
    kubectl delete configmap "${LMEVAL_TASKS_CONFIGMAP}" -n "${KUBE_NAMESPACE}" --ignore-not-found >/dev/null 2>&1 || true
    kubectl create configmap "${LMEVAL_TASKS_CONFIGMAP}" -n "${KUBE_NAMESPACE}" --from-file="${TASKS_DIR}" >/dev/null
    echo "Custom tasks: ${LMEVAL_TASKS_CONFIGMAP} (from ${TASKS_DIR})"
fi

echo "=============================================="
echo "Submit lm-eval Job"
echo "=============================================="
echo "Run name:      ${EVAL_RUN_NAME}"
echo "Job name:      lmeval-${EVAL_RUN_SLUG}"
echo "Served name:   ${VLLM_SERVED_NAME}"
echo "Tasks:         ${LMEVAL_TASKS}"
echo "Limit:         ${LMEVAL_LIMIT}"
echo "Concurrency:   ${LMEVAL_CONCURRENCY}"
echo "Temperature:   ${LMEVAL_TEMPERATURE} (top_p ${LMEVAL_TOP_P})"
echo "Image:         ${LMEVAL_IMAGE} (lm-eval==${LMEVAL_VERSION})"
echo "CPU node:      ${LMEVAL_INSTANCE_TYPE}"
echo "Namespace:     ${KUBE_NAMESPACE}"
echo "Output:        /fsx/data/verl/eval_results/${EVAL_RUN_NAME}/lmeval/results.json"
echo "=============================================="

# Clean any prior job of the same name (Jobs are immutable).
kubectl delete job "lmeval-${EVAL_RUN_SLUG}" -n "${KUBE_NAMESPACE}" --ignore-not-found >/dev/null 2>&1 || true

# Allowlist envsubst: substitute ONLY these template vars so the container-runtime
# shell vars in the manifest args block (${OUT_DIR}, ${BASE_URL}, ${LATEST}, ${i},
# ${LIMIT_ARG}, ${MODELS_URL}) are left untouched for bash to expand at runtime.
SUBST_VARS='${KUBE_NAMESPACE} ${EVAL_RUN_NAME} ${EVAL_RUN_SLUG} ${VLLM_SERVED_NAME} ${LMEVAL_TASKS} ${LMEVAL_LIMIT} ${LMEVAL_CONCURRENCY} ${LMEVAL_TIMEOUT} ${LMEVAL_TEMPERATURE} ${LMEVAL_TOP_P} ${LMEVAL_IMAGE} ${LMEVAL_VERSION} ${LMEVAL_INSTANCE_TYPE} ${LMEVAL_TOKENIZER} ${LMEVAL_MAX_LENGTH} ${LMEVAL_APPLY_CHAT_TEMPLATE} ${LMEVAL_GEN_KWARGS} ${LMEVAL_METADATA_B64} ${LMEVAL_TASKS_CONFIGMAP}'
envsubst "${SUBST_VARS}" < "${MANIFEST}" | kubectl apply -n "${KUBE_NAMESPACE}" -f -

echo ""
echo "Submitted. Follow logs with:"
echo "  kubectl logs -f job/lmeval-${EVAL_RUN_SLUG} -n ${KUBE_NAMESPACE}"
echo "Wait for completion:"
echo "  kubectl wait --for=condition=complete job/lmeval-${EVAL_RUN_SLUG} -n ${KUBE_NAMESPACE} --timeout=6h"
