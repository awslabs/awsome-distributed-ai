#!/bin/bash
# =============================================================================
# DreamZero checkpoint conversion: FSDP DCP shards -> single .pt
#
# RLinf SFT saves sharded FSDP2 checkpoints under
#   {log_path}/{experiment_name}/checkpoints/global_step_<N>/actor/dcp_checkpoint/
# as __<rank>_0.distcp files. The LIBERO simulator eval (eval_embodied_agent.py)
# consumes a single consolidated .pt via runner.ckpt_path. This script converts
# the DCP shards to that .pt using upstream's convert_dcp_to_pt.py.
#
# Env:
#   DCP_PATH    - dcp_checkpoint dir (default derived from CKPT_PATH/EXPERIMENT_NAME/STEP)
#   OUTPUT_PT   - output .pt path (default: <step>/actor/model_state_dict/full_weights.pt)
#   CKPT_PATH, EXPERIMENT_NAME, STEP - used to derive defaults
# =============================================================================
set -euo pipefail
set -x

VENV_NAME="${VENV_NAME:-dreamzero}"
CKPT_PATH="${CKPT_PATH:-/fsx/checkpoints}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-dreamzero-libero-sft}"
# Upstream nests under the config's runner.logger.experiment_name (libero_sft_dreamzero).
INNER_EXPERIMENT="${INNER_EXPERIMENT:-libero_sft_dreamzero}"
STEP="${STEP:-global_step_1}"

CKPT_DIR="${CKPT_PATH}/${EXPERIMENT_NAME}/${INNER_EXPERIMENT}/checkpoints/${STEP}"
DCP_PATH="${DCP_PATH:-${CKPT_DIR}/actor/dcp_checkpoint}"
OUTPUT_PT="${OUTPUT_PT:-${CKPT_DIR}/actor/model_state_dict/full_weights.pt}"

export PYTHONPATH="${PYTHONPATH:-}"
UV_PATH="${UV_PATH:-/opt/venv}"
if [ -f "${UV_PATH}/${VENV_NAME}/bin/activate" ]; then
    source "${UV_PATH}/${VENV_NAME}/bin/activate"
fi
export DREAMZERO_PATH="${DREAMZERO_PATH:-/workspace/DreamZero}"
export PYTHONPATH="${DREAMZERO_PATH}:${PYTHONPATH}"

cd /workspace/RLinf

if [ ! -d "${DCP_PATH}" ]; then
    echo "ERROR: DCP checkpoint dir not found: ${DCP_PATH}"
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PT}")"

echo "=== Converting DCP -> .pt ==="
echo "DCP:    ${DCP_PATH}"
echo "Output: ${OUTPUT_PT}"

python3 rlinf/utils/ckpt_convertor/fsdp_convertor/convert_dcp_to_pt.py \
    --dcp_path "${DCP_PATH}" \
    --output_path "${OUTPUT_PT}"

echo "=== Done ==="
ls -la "${OUTPUT_PT}"
