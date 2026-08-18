#!/usr/bin/env bash
# STATUS: Verified -- trains cleanly (colocated, 2 nodes / 16 GPU H200). GRPO steps run to
#   completion with --colocate + --use-distributed-optimizer + triton MoE runner, and the
#   resulting generation is healthy: rollout/raw_reward 0.578, rollout/repetition_frac 0.0
#   over NUM_ROLLOUT=2 (comparable to the dense 4B run), weight_version uniform / mixed 0.0.
#   The shipped block runs the rollout MoE in pure expert-parallel (moe_tp=1, i.e.
#   EP_SIZE == ROLLOUT_GPUS_PER_ENGINE). Running it tensor-parallel AND expert-parallel at
#   once (moe_tp>1 and moe_ep>1) hits a FlashInfer allreduce-fusion bug in this SGLang build
#   that drops the moe-tp reduce and yields degenerate, zero-reward text; the recipe detects
#   that geometry and disables the fusion (--sglang-enforce-disable-flashinfer-allreduce-fusion)
#   so the combined path trains correctly too. See README.md Known Issues item 2. The
#   disaggregated actor-8 layout (see the parent slime recipe) needs B300 288GB
#   and is UNVERIFIED on H200; colocated 16-GPU with the distributed optimizer is the
#   H200-fitting layout and is the one shipped as default here.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# ============================================================
# miles GRPO Training — Qwen3-30B-A3B MoE on HyperPod EKS
# (Colocated Mode: training and rollout time-share the same 16 GPUs)
#
# This configuration runs the 30B MoE model with:
#   - actor 2 nodes x 8 GPUs = 16 GPU, TP=2 PP=1 CP=1 EP=2
#   - --use-distributed-optimizer shards the 30B optimizer state across the 16 GPU
#     so the static memory fits H200 (141GB); without it the actor OOMs.
#   - rollout time-shares the same 16 GPU (COLOCATE=true), 2 GPU per SGLang engine.
#
# Prerequisites:
#   - Ray cluster deployed via kubernetes/raycluster.yaml (2 workers for 16 GPU)
#   - Model downloaded and converted to torch_dist format
#   - Training data on FSx
#   - source env_vars with the "ALTERNATE: Qwen3-30B-A3B MoE" block in
#     env_vars.colocated.example uncommented
#
# Usage:
#   source env_vars  # with the 30B MoE block (env_vars.colocated.example) active
#   bash recipe/run_grpo_qwen3_30b_a3b.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

# Source environment if not already loaded. Override the file with ENV_FILE,
# e.g.  ENV_FILE=env_vars.disaggregated bash recipe/run_grpo_qwen3_30b_a3b.sh
if [[ -z "${MODEL_LOCAL:-}" ]]; then
    ENV_FILE="${ENV_FILE:-${PROJECT_DIR}/env_vars}"
    echo "[INFO] Sourcing ${ENV_FILE}..."
    source "${ENV_FILE}"
fi

# Validate required variables (same guard as the 4B recipe; without it a missing
# var trips `set -u` with an opaque error deep in the argv build).
for var in MODEL_LOCAL MODEL_DIST PROMPT_DATA CHECKPOINT_DIR MODEL_SCRIPT RM_TYPE \
           COLOCATE TP_SIZE PP_SIZE CP_SIZE EP_SIZE ACTOR_NUM_NODES ACTOR_GPUS_PER_NODE \
           ROLLOUT_NUM_GPUS ROLLOUT_GPUS_PER_ENGINE NUM_ROLLOUT ROLLOUT_BATCH_SIZE \
           N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE MAX_TOKENS_PER_GPU \
           ROLLOUT_MAX_RESPONSE_LEN ROLLOUT_TEMPERATURE LEARNING_RATE SAVE_INTERVAL EVAL_DATA; do
    if [[ -z "${!var:-}" ]]; then
        echo "[ERROR] ${var} is not set. Please configure env_vars."
        exit 1
    fi
done

# --colocate is passed unconditionally below, so COLOCATE=false cannot be honored. Refuse
# rather than warn: continuing would run the colocated layout while the env said otherwise.
if [[ "${COLOCATE:-true}" != "true" ]]; then
    echo "[ERROR] COLOCATE=${COLOCATE}, but this recipe only builds the colocated layout." >&2
    echo "[ERROR] Disaggregating a 30B actor needs B300-class 288GB HBM and a different" >&2
    echo "[ERROR] rollout layout; it is UNVERIFIED here. Set COLOCATE=true." >&2
    exit 1
fi

# SGLang rollout MoE geometry. The recipe passes --rollout-num-gpus-per-engine
# (the engine's tensor-parallel size) and --sglang-expert-parallel-size (EP_SIZE),
# from which SGLang derives moe_ep = EP_SIZE and moe_tp = per-engine GPUs / EP_SIZE.
# EP_SIZE must divide the per-engine GPU count for that split to be integral.
if (( ROLLOUT_GPUS_PER_ENGINE % EP_SIZE != 0 )); then
    echo "[ERROR] EP_SIZE=${EP_SIZE} must divide ROLLOUT_GPUS_PER_ENGINE=${ROLLOUT_GPUS_PER_ENGINE}" >&2
    echo "[ERROR] so the SGLang rollout MoE tensor/expert split is integral." >&2
    exit 1
fi
SGLANG_MOE_TP=$(( ROLLOUT_GPUS_PER_ENGINE / EP_SIZE ))
# When the rollout MoE runs tensor-parallel AND expert-parallel at once (moe_tp>1 and
# moe_ep>1), this SGLang build's FlashInfer allreduce+RMSNorm fusion -- auto-enabled on
# SM90/SM100 -- reduces the post-experts output over the moe-ep group only and drops the
# moe-tp reduce, so each rank keeps a partial sum and generation degenerates to zero
# reward. Disabling the fusion restores the correct two-stage reduce and the combined
# geometry trains cleanly. Pure expert-parallel (moe_tp=1, EP_SIZE == ROLLOUT_GPUS_PER_ENGINE)
# and pure tensor-parallel (moe_ep=1) are unaffected and leave the fusion on. See
# README.md Known Issues item 2.
SGLANG_FUSION_ARGS=()
if (( EP_SIZE > 1 && SGLANG_MOE_TP > 1 )); then
    MOE_GEOM_NOTE="(TP+EP; allreduce fusion disabled)"
    echo "[INFO] Rollout MoE runs moe_tp=${SGLANG_MOE_TP} x moe_ep=${EP_SIZE} (both >1);"
    echo "[INFO] disabling FlashInfer allreduce fusion for this geometry (README Known Issues item 2)."
    SGLANG_FUSION_ARGS+=(--sglang-enforce-disable-flashinfer-allreduce-fusion)
elif (( SGLANG_MOE_TP == 1 )); then
    MOE_GEOM_NOTE="(pure EP)"
else
    MOE_GEOM_NOTE="(pure TP)"
fi

echo "============================================================"
echo "  miles GRPO Training — Qwen3-30B-A3B MoE (Colocated, 16 GPU)"
echo "============================================================"
echo "  Model:             ${MODEL_LOCAL}"
echo "  Megatron ckpt:     ${MODEL_DIST}"
echo "  Training data:     ${PROMPT_DATA}"
echo "  Checkpoints:       ${CHECKPOINT_DIR}/qwen3-30b-a3b-grpo/"
echo "  Actor:             ${ACTOR_NUM_NODES} nodes x ${ACTOR_GPUS_PER_NODE} GPUs"
echo "  Rollout GPUs:      ${ROLLOUT_NUM_GPUS} (${ROLLOUT_GPUS_PER_ENGINE} per engine)"
echo "  Parallelism:       TP=${TP_SIZE} PP=${PP_SIZE} CP=${CP_SIZE} EP=${EP_SIZE}"
echo "  SGLang rollout:    moe_ep=${EP_SIZE} moe_tp=${SGLANG_MOE_TP} ${MOE_GEOM_NOTE}"
echo "  Rollout BS:        ${ROLLOUT_BATCH_SIZE} x ${N_SAMPLES_PER_PROMPT}"
echo "  Global BS:         ${GLOBAL_BATCH_SIZE}"
echo "============================================================"

# Build the train.py flags as a bash ARRAY (not a single string). Each element
# is one argv token, so values are never re-split by a shell. The array is
# expanded into the `ray job submit -- ...` argv below; MODEL_ARGS itself is
# expanded inside recipe/launcher/grpo_launch.sh, in the same shell that sources
# the miles model script. See that launcher for why this avoids the shell
# escaping trap that a `-- bash -c "...${MODEL_ARGS[@]}..."` string would hit.
#
# When RM_TYPE=remote_rm, point miles at the CPU-hosted reward Service via
# --rm-url (see kubernetes/reward-service.yaml). Otherwise scoring is in-process.
RM_ARGS=(--rm-type "${RM_TYPE}")
if [ "${RM_TYPE}" = "remote_rm" ]; then
    if [ -z "${RM_URL:-}" ]; then
        echo "[ERROR] RM_TYPE=remote_rm but RM_URL is not set. Configure it in env_vars."
        exit 1
    fi
    RM_ARGS+=(--rm-url "${RM_URL}")
    echo "  Reward:         remote_rm @ ${RM_URL}"
fi

TRAIN_ARGS=(
    --hf-checkpoint "${MODEL_LOCAL}"
    --ref-load "${MODEL_DIST}"
    --load "${CHECKPOINT_DIR}/qwen3-30b-a3b-grpo/"
    --save "${CHECKPOINT_DIR}/qwen3-30b-a3b-grpo/"
    --save-interval "${SAVE_INTERVAL}"

    --prompt-data "${PROMPT_DATA}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle

    "${RM_ARGS[@]}"

    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT:-1}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"

    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE}"
    --balance-data

    --eval-interval 10
    --eval-prompt-data aime "${EVAL_DATA}"
    --n-samples-per-eval-prompt 4
    --eval-max-response-len 16384
    --eval-top-p 1

    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size "${PP_SIZE}"
    --context-parallel-size "${CP_SIZE}"
    --expert-model-parallel-size "${EP_SIZE}"
    --expert-tensor-parallel-size 1
    --sequence-parallel

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"

    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28

    --optimizer adam
    --lr "${LEARNING_RATE}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98

    --actor-num-nodes "${ACTOR_NUM_NODES}"
    --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}"
    # Colocated on 16 GPU: actor and rollout time-share the same GPUs. Required
    # for 30B on H200 -- with all 16 GPU behind the actor, --use-distributed-optimizer
    # shards the 30B optimizer state so static memory fits 141GB. Confining the
    # actor to 8 GPU (a single node) OOMs (30B static ~121GB/GPU on 8-way).
    --colocate
    --use-distributed-optimizer
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
    --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"

    # 0.75 is the value that completed on H200. Colocated, the rollout engine shares each GPU
    # with the actor, so the usual single-tenant fractions (0.8-0.85) leave too little room.
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.75}"
    # MoE online weight update on SGLang 0.5.12+ requires the triton runner: the
    # default flashinfer MoE runner is incompatible with SLIME/miles's in-place
    # weight update, so the engine must serve with --sglang-moe-runner-backend
    # triton and expert parallelism must be declared to SGLang explicitly.
    --sglang-moe-runner-backend triton
    --sglang-expert-parallel-size "${EP_SIZE}"
    # Disable the FlashInfer allreduce+RMSNorm fusion only when moe_tp>1 and moe_ep>1
    # (empty otherwise); the fusion mis-handles that combined path in this build.
    ${SGLANG_FUSION_ARGS[@]+"${SGLANG_FUSION_ARGS[@]}"}
    # Lowercase only: this reaches uvicorn's log_level, whose LOG_LEVELS dict has no "WARN"
    # key, and the KeyError kills the rollout server before it binds. See docs/PORT_NOTES.md.
    --sglang-log-level warning
    # Careful when adding --sglang-* flags: miles registers them from SGLang's live
    # ServerArgs with ignore_unknown_args, so a flag SGLang has since removed is silently
    # accepted and does nothing. Check it against the SGLang version in the base image.
)

# Optional extra train.py flags injected via EXTRA_TRAIN_ARGS, same mechanism as
# the 4B recipe (e.g. --use-tensorboard). Any flag taking a module path must be a
# DOTTED module path, not "file.py:func" -- load_function does rpartition('.') +
# import_module, so the slash form fails with ModuleNotFoundError at the loss
# forward. The miles package is on PYTHONPATH (/root/miles) below.
EXTRA_TRAIN_ARGS_ARR=()
if [ -n "${EXTRA_TRAIN_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_TRAIN_ARGS_ARR=(${EXTRA_TRAIN_ARGS})
    echo "  Extra args:     ${EXTRA_TRAIN_ARGS}"
fi
# ${arr[@]+...} guards the empty-array + `set -u` case on bash < 4.4 (macOS 3.2).
TRAIN_ARGS+=(${EXTRA_TRAIN_ARGS_ARR[@]+"${EXTRA_TRAIN_ARGS_ARR[@]}"})

# Submit via Ray job API.
#
# The entrypoint after `--` is `bash grpo_launch.sh <flags>` (plain argv tokens,
# no shell array crosses the ray boundary). --working-dir uploads the launcher
# to the Ray workers; the miles code itself is already in the image at
# /root/miles (on PYTHONPATH below). MODEL_SCRIPT is forwarded so the launcher
# can source the right model definition.
#
# --entrypoint-resources '{"gpu_node": 0.001}' pins the driver to a GPU worker
# (miles imports mooncake / libcuda at module load; the head is a non-GPU pod).
# CUDA_DEVICE_MAX_CONNECTIONS=1 is required by Megatron for TP>1 (30B is TP=2).
# HF_TOKEN is NOT set here: it is injected into the pod env from the k8s Secret in
# raycluster.yaml, so it never lands in the Ray GCS runtime-env.
echo "[INFO] Submitting Ray job for MoE GRPO training..."

ray job submit \
    --address="http://127.0.0.1:8265" \
    --entrypoint-resources '{"gpu_node": 0.001}' \
    --working-dir "${SCRIPT_DIR}/launcher" \
    --runtime-env-json="{
        \"env_vars\": {
            \"PYTHONPATH\": \"/root/Megatron-LM:/root/miles\",
            \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
            \"MODEL_SCRIPT\": \"${MODEL_SCRIPT}\",
            \"TOKENIZERS_PARALLELISM\": \"false\",
            \"NCCL_DEBUG\": \"WARN\",
            \"FI_PROVIDER\": \"efa\",
            \"FI_EFA_USE_DEVICE_RDMA\": \"1\",
            \"TENSORBOARD_DIR\": \"${TENSORBOARD_DIR:-}\"
        }
    }" \
    -- bash grpo_launch.sh "${TRAIN_ARGS[@]}"

echo "[INFO] Job submitted. Monitor at http://localhost:8265"
