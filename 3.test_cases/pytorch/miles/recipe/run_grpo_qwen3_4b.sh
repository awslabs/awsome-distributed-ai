#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# ============================================================
# miles GRPO Training — Qwen3-4B on HyperPod EKS
#
# Submits a Ray job that runs GRPO with Megatron-LM for training and SGLang for
# rollout. COLOCATE picks how the two share the cluster, and with it how weights
# reach the rollout engines after every training step:
#
#   COLOCATE=true   one GPU pool, time-shared. Weights move by CUDA IPC.
#   COLOCATE=false  two GPU pools. Weights move by NCCL broadcast, over EFA when
#                   the pools are on different nodes.
#
# Both layouts are exercised by this recipe; see the COLOCATE block below for the
# sizing rules each one imposes.
#
# Prerequisites:
#   - Ray cluster deployed via kubernetes/raycluster.yaml
#   - Model downloaded and converted to torch_dist format
#   - Training data on FSx
#   - source env_vars before running
#
# Usage:
#   source env_vars
#   bash recipe/run_grpo_qwen3_4b.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

# Source environment if not already loaded. Override the file with ENV_FILE,
# e.g.  ENV_FILE=env_vars.disaggregated bash recipe/run_grpo_qwen3_4b.sh
if [[ -z "${MODEL_LOCAL:-}" ]]; then
    ENV_FILE="${ENV_FILE:-${PROJECT_DIR}/env_vars}"
    echo "[INFO] Sourcing ${ENV_FILE}..."
    source "${ENV_FILE}"
fi

# Validate required variables
for var in MODEL_LOCAL MODEL_DIST PROMPT_DATA CHECKPOINT_DIR MODEL_SCRIPT RM_TYPE \
           COLOCATE TP_SIZE PP_SIZE CP_SIZE EP_SIZE ACTOR_NUM_NODES ACTOR_GPUS_PER_NODE \
           ROLLOUT_NUM_GPUS ROLLOUT_GPUS_PER_ENGINE NUM_ROLLOUT ROLLOUT_BATCH_SIZE \
           N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE MAX_TOKENS_PER_GPU ROLLOUT_MAX_RESPONSE_LEN \
           ROLLOUT_TEMPERATURE LEARNING_RATE SAVE_INTERVAL EVAL_DATA; do
    if [[ -z "${!var:-}" ]]; then
        echo "[ERROR] ${var} is not set. Please configure env_vars."
        exit 1
    fi
done

# COLOCATE selects the weight sync path, which is the whole reason this variable exists.
# miles picks the implementation from --colocate alone (backends/megatron_utils/actor.py):
#
#   --colocate present  UpdateWeightFromTensor      actor and rollout share devices; weights
#                                                   move by CUDA IPC handle
#   --colocate absent   UpdateWeightFromDistributed actor and rollout own separate devices;
#                                                   weights move by NCCL broadcast, over EFA
#                                                   when the two pools sit on different nodes
#                                                   (--update-weight-transfer-mode defaults
#                                                   to "broadcast")
#
# Accept only the two spellings. Under `set -u` an unset COLOCATE aborts, but "True", "1" and
# "yes" would otherwise fall through to the disaggregated branch and pick a layout nobody
# asked for -- the same silent disagreement as printing "Colocated: false" while running the
# colocated layout, just pointing the other way.
case "${COLOCATE}" in
    true|false) ;;
    *) echo "[ERROR] COLOCATE must be exactly 'true' or 'false', got '${COLOCATE}'." >&2
       echo "[ERROR] Values like True/1/yes would silently select a layout you did not ask for." >&2
       exit 1 ;;
esac

# Arithmetic and -ne on a non-numeric value do not abort the script: the comparison itself
# errors, `if` reads that as false, and the layout check below passes without having run. A
# check that cannot run is worse than no check, because it reads as a pass. $((...)) also
# re-evaluates its operands as expressions, so a non-numeric value is an injection surface.
for _v in ACTOR_NUM_NODES ACTOR_GPUS_PER_NODE ROLLOUT_NUM_GPUS ROLLOUT_GPUS_PER_ENGINE; do
    if ! [[ "${!_v}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] ${_v} must be a non-negative integer, got '${!_v}'." >&2
        exit 1
    fi
done
unset _v

# Engines must tile the rollout pool exactly, in either layout. A remainder leaves GPUs with
# no engine, or asks an engine for a shard that does not exist; neither shows up at submit
# time, so check it here where the numbers are still in view.
if [[ "${ROLLOUT_GPUS_PER_ENGINE}" -le 0 ]] \
   || [[ $((ROLLOUT_NUM_GPUS % ROLLOUT_GPUS_PER_ENGINE)) -ne 0 ]]; then
    echo "[ERROR] ROLLOUT_GPUS_PER_ENGINE (${ROLLOUT_GPUS_PER_ENGINE}) must be a positive" >&2
    echo "[ERROR] divisor of ROLLOUT_NUM_GPUS (${ROLLOUT_NUM_GPUS})." >&2
    exit 1
fi

# Resolve the KV-pool fraction once. Two readers of the same default drift apart: fix one and
# the banner starts describing a different run than the argv does. Colocated keeps the value
# this recipe has always used, so an existing colocated invocation renders an unchanged argv.
MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.8}"

# An array, not a string: the disaggregated case must expand to ZERO argv tokens, and an empty
# string would reach argparse as a stray positional. Same property the recipe already relies on
# for EXTRA_TRAIN_ARGS_ARR.
COLOCATE_ARGS=()
[[ "${COLOCATE}" == "true" ]] && COLOCATE_ARGS=(--colocate)

ACTOR_GPUS=$((ACTOR_NUM_NODES * ACTOR_GPUS_PER_NODE))
TOTAL_GPUS=$((ACTOR_GPUS + ROLLOUT_NUM_GPUS))

# Holds in either layout: the trainer alone cannot exceed the cluster. Checked before the
# per-layout rules so an impossible actor size is reported as such, rather than surfacing as
# whichever layout-specific inequality happens to trip first.
if [[ -n "${CLUSTER_GPUS:-}" ]] && [[ "${ACTOR_GPUS}" -gt "${CLUSTER_GPUS}" ]]; then
    echo "[ERROR] actor needs ${ACTOR_NUM_NODES} x ${ACTOR_GPUS_PER_NODE} = ${ACTOR_GPUS} GPUs," >&2
    echo "[ERROR] more than CLUSTER_GPUS=${CLUSTER_GPUS}." >&2
    exit 1
fi

if [[ "${COLOCATE}" == "true" ]]; then
    # Sharing devices means the rollout count IS the actor count. A mismatch would place
    # engines on a different number of GPUs than the trainer holds.
    if [[ "${ROLLOUT_NUM_GPUS}" -ne "${ACTOR_GPUS}" ]]; then
        echo "[ERROR] COLOCATE=true shares devices, so ROLLOUT_NUM_GPUS (${ROLLOUT_NUM_GPUS})" >&2
        echo "[ERROR] must equal the actor GPU count (${ACTOR_NUM_NODES} x ${ACTOR_GPUS_PER_NODE} = ${ACTOR_GPUS})." >&2
        exit 1
    fi
else
    # Separate pools must both fit, and over-subscribing does not fail loudly: Ray waits on a
    # placement group that never becomes ready, which reads as a hang rather than as a
    # misconfiguration. CLUSTER_GPUS is optional because only the caller knows the cluster;
    # when it is set, refuse here instead. Note that actor == rollout is the INTENDED shape on
    # a 2-node 8-GPU cluster (8 + 8 = 16), not a mistake to warn about.
    if [[ -n "${CLUSTER_GPUS:-}" ]] && [[ "${TOTAL_GPUS}" -gt "${CLUSTER_GPUS}" ]]; then
        echo "[ERROR] COLOCATE=false needs actor ${ACTOR_GPUS} + rollout ${ROLLOUT_NUM_GPUS}" >&2
        echo "[ERROR] = ${TOTAL_GPUS} GPUs, more than CLUSTER_GPUS=${CLUSTER_GPUS}." >&2
        echo "[ERROR] Ray would wait forever on an unschedulable placement group." >&2
        exit 1
    fi
fi

echo "============================================================"
echo "  miles GRPO Training — Qwen3-4B"
echo "============================================================"
echo "  Model:          ${MODEL_LOCAL}"
echo "  Megatron ckpt:  ${MODEL_DIST}"
echo "  Training data:  ${PROMPT_DATA}"
echo "  Checkpoints:    ${CHECKPOINT_DIR}/qwen3-4b-grpo/"
echo "  Nodes:          ${ACTOR_NUM_NODES} x ${ACTOR_GPUS_PER_NODE} GPUs"
if [[ "${COLOCATE}" == "true" ]]; then
    echo "  Weight sync:    colocated / CUDA IPC (UpdateWeightFromTensor)"
    echo "  GPUs:           ${ACTOR_GPUS} shared by actor and rollout"
else
    echo "  Weight sync:    disaggregated / NCCL broadcast (UpdateWeightFromDistributed)"
    echo "  GPUs:           actor ${ACTOR_GPUS} + rollout ${ROLLOUT_NUM_GPUS} = ${TOTAL_GPUS}"
fi
# The static fraction bounds each engine's KV pool. Colocated must leave room for the trainer
# on the same device; disaggregated owns its GPUs and can take more. Print the value that the
# flag below actually receives, so the log never describes a run that did not happen.
echo "  Mem fraction:   ${MEM_FRACTION}"
echo "  Rollout BS:     ${ROLLOUT_BATCH_SIZE} x ${N_SAMPLES_PER_PROMPT}"
echo "  Global BS:      ${GLOBAL_BATCH_SIZE}"
echo "  Num rollouts:   ${NUM_ROLLOUT}"
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

# Optional extra train.py flags, injected without editing this recipe. Set the
# EXTRA_TRAIN_ARGS env var to a whitespace-separated list of flags in env_vars
# (or on the command line) and they are appended verbatim to the train.py argv.
# This is the supported extension point for flags the baseline recipe does not
# set — e.g. observability (--use-tensorboard) or LR scheduling
# (--lr-decay-style ...). NOTE: any flag that takes a module path (e.g. a custom
# reward or callback) must be a DOTTED module path (load_function does
# rpartition('.') + import_module); the "file.py:func" form fails with
# ModuleNotFoundError. Word-splitting here is intentional so a single env var can
# carry several flags; values containing spaces are not supported (none of the
# intended flags need them). The array keeps each token separate across the Ray
# job boundary, the same shell-safety property the rest of this recipe relies on.
EXTRA_TRAIN_ARGS_ARR=()
if [ -n "${EXTRA_TRAIN_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_TRAIN_ARGS_ARR=(${EXTRA_TRAIN_ARGS})
    echo "  Extra args:     ${EXTRA_TRAIN_ARGS}"
fi

TRAIN_ARGS=(
    --hf-checkpoint "${MODEL_LOCAL}"
    --ref-load "${MODEL_DIST}"
    --load "${CHECKPOINT_DIR}/qwen3-4b-grpo/"
    --save "${CHECKPOINT_DIR}/qwen3-4b-grpo/"
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
    --n-samples-per-eval-prompt 8
    --eval-max-response-len 16384
    --eval-top-p 1

    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size "${PP_SIZE}"
    --context-parallel-size "${CP_SIZE}"
    --expert-model-parallel-size "${EP_SIZE}"
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
    ${COLOCATE_ARGS[@]+"${COLOCATE_ARGS[@]}"}
    # Colocated: actor and rollout share the same GPUs, so rollout-num-gpus should
    # equal the actor GPU count (ACTOR_NUM_NODES x ACTOR_GPUS_PER_NODE). On 1 node
    # this is 8; on 2 nodes 16. Set it explicitly so scaling nodes does not silently
    # leave the rollout at a stale GPU count.
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
    --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"

    --sglang-mem-fraction-static "${MEM_FRACTION}"
    # Lowercase only: this reaches uvicorn's log_level, whose LOG_LEVELS dict has no "WARN"
    # key, and the KeyError kills the rollout server before it binds -- training then hangs
    # on a health check that never passes. See docs/PORT_NOTES.md.
    --sglang-log-level warning

    # Flags injected via the EXTRA_TRAIN_ARGS env var. The shipped default is just
    # observability (--use-tensorboard), which does not change the loss; set it to
    # empty for a bit-identical baseline, or add flags. See the block above.
    # The ${arr[@]+"${arr[@]}"} form expands to nothing (not an "unbound variable"
    # error) when the array is empty under `set -u` on bash < 4.4 (e.g. macOS 3.2).
    ${EXTRA_TRAIN_ARGS_ARR[@]+"${EXTRA_TRAIN_ARGS_ARR[@]}"}
)

# Submit via Ray job API.
#
# The entrypoint after `--` is `bash grpo_launch.sh <flags>` (plain argv tokens,
# no shell array crosses the ray boundary). --working-dir uploads the launcher
# to the Ray workers; the miles code itself is already in the image at
# /root/miles (on PYTHONPATH below). MODEL_SCRIPT is forwarded so the launcher
# can source the right model definition.
#
# --entrypoint-resources '{"gpu_node": 0.001}' pins the Ray job DRIVER to a GPU
# worker. miles imports mooncake (libcuda-dependent) at module load, and the head
# is a non-GPU pod, so a head-scheduled driver dies with a libcuda import error.
# The 0.001 fractional request lands the driver on a worker without consuming a
# whole GPU (does not disturb the colocated placement group). HF_TOKEN is NOT set
# here: it is injected into the pod env from the k8s Secret in raycluster.yaml, so
# it never lands in the Ray GCS runtime-env (visible via the dashboard API).
echo "[INFO] Submitting Ray job..."

ray job submit \
    --address="http://127.0.0.1:8265" \
    --entrypoint-resources '{"gpu_node": 0.001}' \
    --working-dir "${SCRIPT_DIR}/launcher" \
    --runtime-env-json="{
        \"env_vars\": {
            \"PYTHONPATH\": \"/root/Megatron-LM:/root/miles\",
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
