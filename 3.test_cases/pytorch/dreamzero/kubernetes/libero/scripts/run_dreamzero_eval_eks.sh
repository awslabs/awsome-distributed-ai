#!/bin/bash
# =============================================================================
# DreamZero LIBERO simulator eval launcher (single-node, GPU).
#
# Runs upstream examples/embodiment/eval_embodied_agent.py against a trained
# DreamZero checkpoint (full_weights.pt) in the LIBERO Spatial simulator,
# reporting success_once and (optionally) saving in-sim rollout videos.
#
# Env:
#   VENV_NAME        - venv (default: dreamzero)
#   CONFIG_NAME      - eval config (default: libero_spatial_eval_dreamzero_14b)
#   CKPT_PT          - full_weights.pt to evaluate (runner.ckpt_path)
#   METADATA_PATH    - libero_sim metadata.json (default: /fsx/models/metadata-libero.json)
#   TOKENIZER_PATH   - umt5-xxl (default: /fsx/models/umt5-xxl)
#   MODEL_PATH       - DreamZero-DROID backbone dir (for component build; default staged)
#   LOG_DIR          - eval output (default: /fsx/checkpoints/dreamzero-libero-eval)
#   SAVE_VIDEO       - "True"/"False" save in-sim rollout video (default: True)
#   HYDRA_OVERRIDES  - extra Hydra args
# =============================================================================
set -euo pipefail
set -x

VENV_NAME="${VENV_NAME:-dreamzero}"
CONFIG_NAME="${CONFIG_NAME:-libero_spatial_eval_dreamzero_14b}"
CKPT_PT="${CKPT_PT:?set CKPT_PT to the full_weights.pt to evaluate}"
METADATA_PATH="${METADATA_PATH:-/fsx/models/metadata-libero.json}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/fsx/models/umt5-xxl}"
MODEL_PATH="${MODEL_PATH:-/fsx/models/DreamZero-DROID}"
LOG_DIR="${LOG_DIR:-/fsx/checkpoints/dreamzero-libero-eval}"
SAVE_VIDEO="${SAVE_VIDEO:-True}"

export PYTHONPATH="${PYTHONPATH:-}"
UV_PATH="${UV_PATH:-/opt/venv}"
if [ -f "${UV_PATH}/${VENV_NAME}/bin/activate" ]; then
    echo "Activating venv: ${VENV_NAME}"
    source "${UV_PATH}/${VENV_NAME}/bin/activate"
elif [ -f "/usr/local/bin/switch_env" ]; then
    source switch_env "${VENV_NAME}"
else
    echo "WARNING: No venv found for ${VENV_NAME}, using system Python"
fi

export DREAMZERO_PATH="${DREAMZERO_PATH:-/workspace/DreamZero}"
export PYTHONPATH="${DREAMZERO_PATH}:${PYTHONPATH}"
export EMBODIED_PATH="/workspace/RLinf/examples/embodiment"
# LIBERO simulator requires headless GL; osmesa is the safe EKS default.
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

cd /workspace/RLinf

# --- Stage the 14B eval config into the embodiment config dir ---
# eval_embodied_agent.py runs with `--config-path config`, i.e.
# /workspace/RLinf/examples/embodiment/config/. Upstream only ships a 5B eval
# config there; our 14B variant is mounted (via the dreamzero-eval-config
# ConfigMap) at /opt/eval-config/ and copied into the embodiment config dir here.
# We copy a single file (NOT mount a ConfigMap over the dir) so the upstream
# config groups (env/, model/, training_backend/, weight_syncer/) remain visible.
EVAL_CONFIG_SRC="${EVAL_CONFIG_SRC:-/opt/eval-config}"
EMBODIED_CONFIG_DIR="/workspace/RLinf/examples/embodiment/config"
if [ -f "${EVAL_CONFIG_SRC}/${CONFIG_NAME}.yaml" ]; then
    echo "Staging eval config: ${EVAL_CONFIG_SRC}/${CONFIG_NAME}.yaml -> ${EMBODIED_CONFIG_DIR}/"
    cp "${EVAL_CONFIG_SRC}/${CONFIG_NAME}.yaml" "${EMBODIED_CONFIG_DIR}/${CONFIG_NAME}.yaml"
else
    echo "No mounted eval config at ${EVAL_CONFIG_SRC}/${CONFIG_NAME}.yaml;"
    echo "expecting ${EMBODIED_CONFIG_DIR}/${CONFIG_NAME}.yaml to already exist."
fi

if [ ! -f "${CKPT_PT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT_PT}"
    echo "Convert the DCP checkpoint to .pt first (convert-checkpoint.yaml)."
    exit 1
fi
mkdir -p "${LOG_DIR}"

HYDRA_ARGS="runner.only_eval=True"
HYDRA_ARGS="${HYDRA_ARGS} runner.ckpt_path=${CKPT_PT}"
HYDRA_ARGS="${HYDRA_ARGS} runner.logger.log_path=${LOG_DIR}"
HYDRA_ARGS="${HYDRA_ARGS} actor.model.tokenizer_path=${TOKENIZER_PATH}"
HYDRA_ARGS="${HYDRA_ARGS} actor.model.model_path=${MODEL_PATH}"
# metadata_json_path is commented out in the config struct -> add with '+'.
HYDRA_ARGS="${HYDRA_ARGS} +actor.model.metadata_json_path=${METADATA_PATH}"
HYDRA_ARGS="${HYDRA_ARGS} env.eval.video_cfg.save_video=${SAVE_VIDEO}"
if [ -n "${HYDRA_OVERRIDES:-}" ]; then
    HYDRA_ARGS="${HYDRA_ARGS} ${HYDRA_OVERRIDES}"
fi

echo "Eval Hydra args: ${HYDRA_ARGS}"
# shellcheck disable=SC2086
python3 examples/embodiment/eval_embodied_agent.py \
    --config-path config \
    --config-name "${CONFIG_NAME}" \
    ${HYDRA_ARGS}
