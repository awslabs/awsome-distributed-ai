#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# RLinf Training Launch Script for Amazon EKS
#
# Generic launcher that works with ANY RLinf embodiment config.
# The upstream multi-venv pattern is preserved: set VENV_NAME to select the
# model's Python environment, then CONFIG_NAME to select the Hydra config.
#
# Expects:
#   - Container built from the two-stage Dockerfile (upstream RLinf + EFA)
#   - Pre-trained model weights at MODEL_PATH (on shared storage)
#   - ManiSkill/simulator assets available (via model-download job or baked in)
#   - Shared storage at /fsx (FSx for Lustre PVC)
#   - EFA environment variables set via pod spec
#
# Key environment variables (set via K8s manifest env):
#   CONFIG_NAME    - Hydra config name (e.g., maniskill_ppo_openvla_quickstart)
#   VENV_NAME      - Python venv to activate (e.g., openvla, openvla-oft, openpi)
#   MODEL_PATH     - Path to pre-trained model weights on shared storage
#   CKPT_PATH      - Path to write checkpoints
#   NUM_GPUS       - GPUs per node
#   NUM_NODES      - Number of nodes
# =============================================================================
set -euo pipefail
set -x

# --- Configuration (override via environment variables) ---
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rlinf-eks}"
CONFIG_NAME="${CONFIG_NAME:-maniskill_ppo_openvla_quickstart}"
VENV_NAME="${VENV_NAME:-openvla}"

MODEL_PATH="${MODEL_PATH:-/fsx/models/openvla-7b-rlvla-warmup}"
CKPT_PATH="${CKPT_PATH:-/fsx/checkpoints}"

NUM_GPUS="${NUM_GPUS:-8}"
NUM_NODES="${NUM_NODES:-1}"

# Component placement string (e.g., "0-7" for 8 GPUs)
GPU_RANGE="0-$((NUM_GPUS - 1))"
COMPONENT_PLACEMENT="${COMPONENT_PLACEMENT:-actor,env,rollout: ${GPU_RANGE}}"

# --- Activate the correct venv ---
# Upstream RLinf images use multi-venv pattern with switch_env utility.
# Each model (openvla, openvla-oft, openpi, gr00t, etc.) has its own venv.
UV_PATH="${UV_PATH:-/opt/venv}"

# Pre-set variables that venv activate scripts may reference but are not
# guaranteed to exist in container environments (avoids "unbound variable"
# errors under set -u).
export PYTHONPATH="${PYTHONPATH:-}"

if [ -f "${UV_PATH}/${VENV_NAME}/bin/activate" ]; then
    echo "Activating venv: ${VENV_NAME}"
    source "${UV_PATH}/${VENV_NAME}/bin/activate"
elif [ -f "/usr/local/bin/switch_env" ]; then
    echo "Activating venv via switch_env: ${VENV_NAME}"
    source switch_env "${VENV_NAME}"
else
    echo "WARNING: No venv found for ${VENV_NAME}, using system Python"
fi

echo "Python: $(which python3)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"

# --- Environment (defaults set in Dockerfile, override via pod spec) ---
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

# Ensure libcuda.so is discoverable by subprocesses (Ray env offload workers).
# The NVIDIA runtime mounts it at /usr/lib64 but that may not be in LD_LIBRARY_PATH.
export LD_LIBRARY_PATH="/usr/lib64:${LD_LIBRARY_PATH:-}"

# ManiSkill / MuJoCo headless rendering
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"

# EFA (set in pod spec, but provide defaults)
export FI_PROVIDER="${FI_PROVIDER:-efa}"
export FI_EFA_USE_DEVICE_RDMA="${FI_EFA_USE_DEVICE_RDMA:-1}"
export FI_EFA_FORK_SAFE="${FI_EFA_FORK_SAFE:-1}"

# --- Link simulator assets if available ---
# The upstream image provides link_assets for ManiSkill/SAPIEN symlinks
if [ -x "/usr/local/bin/link_assets" ]; then
    link_assets
fi

# --- Pre-step: Verify model weights ---
if [ -n "${MODEL_PATH}" ] && [ "${MODEL_PATH}" != "none" ]; then
    if [ ! -f "$MODEL_PATH/config.json" ] && [ ! -f "$MODEL_PATH/model.safetensors" ]; then
        echo "ERROR: Model weights not found at $MODEL_PATH"
        echo "Expected config.json or model.safetensors. Run the model-download job first."
        exit 1
    fi
fi

# Clear stale Python bytecode from previous runs (FSx is persistent storage)
find "$CKPT_PATH" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Clear HuggingFace transformers dynamic module cache
rm -rf /root/.cache/huggingface/modules/transformers_modules/ 2>/dev/null || true

# --- Launch training ---
echo "=== Launching RLinf training ==="
echo "Config: ${CONFIG_NAME}"
echo "Venv:   ${VENV_NAME}"
echo "Model:  ${MODEL_PATH}"
echo "GPUs:   ${NUM_GPUS} x ${NUM_NODES} nodes"

cd /workspace/RLinf

# Set EMBODIED_PATH for Hydra config interpolation (used in upstream configs
# to resolve relative paths to examples/embodiment/).
export EMBODIED_PATH="${EMBODIED_PATH:-/workspace/RLinf/examples/embodiment}"

# Build Hydra override args.
# Start with model paths and cluster config, then append any user-supplied overrides.
# HYDRA_OVERRIDES env var allows callers (e.g., validation harness) to inject
# additional overrides like "runner.max_steps=1" without modifying this script.
HYDRA_ARGS=""
if [ -n "${MODEL_PATH}" ] && [ "${MODEL_PATH}" != "none" ]; then
    HYDRA_ARGS="actor.model.model_path=${MODEL_PATH} rollout.model.model_path=${MODEL_PATH}"
fi
HYDRA_ARGS="${HYDRA_ARGS} cluster.num_nodes=${NUM_NODES}"

# Append user-supplied overrides (e.g., HYDRA_OVERRIDES="runner.max_steps=1")
if [ -n "${HYDRA_OVERRIDES:-}" ]; then
    echo "Extra Hydra overrides: ${HYDRA_OVERRIDES}"
    HYDRA_ARGS="${HYDRA_ARGS} ${HYDRA_OVERRIDES}"
fi

# Support YAML config override file for complex keys (e.g., component_placement
# with commas in the key name that Hydra CLI cannot parse).
# Set HYDRA_CONFIG_FILE to a YAML file path; its contents will be patched into
# the base config file before launching training. The container is ephemeral,
# so in-place modification is safe.
if [ -n "${HYDRA_CONFIG_FILE:-}" ] && [ -f "${HYDRA_CONFIG_FILE}" ]; then
    CONFIG_FILE="examples/embodiment/config/${CONFIG_NAME}.yaml"
    echo "Patching config with override file: ${HYDRA_CONFIG_FILE}"
    python3 -c "
import yaml, sys
with open('${CONFIG_FILE}') as f:
    base = yaml.safe_load(f)
with open('${HYDRA_CONFIG_FILE}') as f:
    override = yaml.safe_load(f)

def deep_merge(base, override):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v

deep_merge(base, override)
with open('${CONFIG_FILE}', 'w') as f:
    yaml.dump(base, f, default_flow_style=False, sort_keys=False)
print(f'Patched {len(override)} top-level keys into ${CONFIG_FILE}')
"
fi

LOG_DIR="${CKPT_PATH}/${EXPERIMENT_NAME}"
HYDRA_ARGS="${HYDRA_ARGS} runner.logger.log_path=${LOG_DIR}"

echo "Hydra args: ${HYDRA_ARGS}"

# Call train_embodied_agent.py directly (not run_embodiment.sh) so we can pass
# arbitrary Hydra overrides. run_embodiment.sh does not forward extra args.
# --config-path is relative to the script's directory (examples/embodiment/),
# so use just "config" not the full path from repo root.
# shellcheck disable=SC2086
python3 examples/embodiment/train_embodied_agent.py \
    --config-path config \
    --config-name "${CONFIG_NAME}" \
    ${HYDRA_ARGS}
