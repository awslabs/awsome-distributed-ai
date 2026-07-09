#!/bin/bash
# Pi0 (LeRobot) — Training Entrypoint for HyperPod EKS
#
# This script runs inside the training container on each worker pod.
# It expects the following env vars (set by the PyTorchJob manifest):
#   DATASET_REPO_ID   - LeRobot dataset repo_id (e.g. droid_local)
#   DATASET_PATH      - Path to the dataset on FSx (e.g. /data/datasets/droid_100_train)
#   CHECKPOINT_PATH   - Where to save checkpoints (e.g. /data/runs/pi0-droid)
#   HF_TOKEN          - HuggingFace token (for base model download)
#   MAX_STEPS         - Training steps (default: 20000)
#   BATCH_SIZE        - Per-GPU batch size (default: 4)
#   LEARNING_RATE     - LR (default: 2.5e-5)
#   ACTION_HORIZON    - Action chunk length (default: 50)
#   SAVE_INTERVAL     - Checkpoint save interval (default: 2000)
#   NUM_GPUS          - GPUs per node (default: 8)

set -euo pipefail

echo "============================================"
echo " Pi0 (LeRobot) — HyperPod EKS Training"
echo "============================================"
echo "  Host:     $(hostname)"
echo "  GPUs:     ${NUM_GPUS:-8}"
echo "  Dataset:  ${DATASET_REPO_ID:-droid_local}"
echo "  Steps:    ${MAX_STEPS:-20000}"
echo "  Batch:    ${BATCH_SIZE:-4} per GPU"
echo "============================================"

# --- HuggingFace login ---
if [ -n "${HF_TOKEN:-}" ]; then
    echo "[setup] Logging into HuggingFace..."
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" 2>/dev/null || true
fi

# --- Configure paths ---
DATASET_REPO_ID="${DATASET_REPO_ID:-droid_local}"
DATASET_PATH="${DATASET_PATH:-/data/datasets/droid_100_train}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/data/runs/pi0-droid}"
MAX_STEPS="${MAX_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-2.5e-5}"
ACTION_HORIZON="${ACTION_HORIZON:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
NUM_GPUS="${NUM_GPUS:-8}"

# --- Set up LeRobot dataset cache ---
# LeRobot's lerobot-train looks up datasets in HF_LEROBOT_HOME.
# Symlink the FSx dataset path so the CLI finds it as a local repo.
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/lerobot-cache}"
mkdir -p "${HF_LEROBOT_HOME}"
ln -snf "${DATASET_PATH}" "${HF_LEROBOT_HOME}/${DATASET_REPO_ID}"
echo "[setup] Symlinked dataset: ${HF_LEROBOT_HOME}/${DATASET_REPO_ID} -> ${DATASET_PATH}"

# --- Apply local dataset patch ---
# Prevents LeRobot from making Hub API calls for local-only repo_ids
export HF_HUB_OFFLINE=1
python /opt/pi0-lerobot/src/lerobot_local_patch.py 2>/dev/null || true

# --- Create output directory ---
mkdir -p "${CHECKPOINT_PATH}"

# --- Launch training ---
echo "[train] Starting accelerate launch with ${NUM_GPUS} GPUs..."

accelerate launch \
    --multi_gpu \
    --num_processes="${NUM_GPUS}" \
    --mixed_precision=bf16 \
    $(python -c "import lerobot; import os; print(os.path.join(os.path.dirname(lerobot.__file__), '..', 'scripts', 'train.py'))" 2>/dev/null || which lerobot-train) \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_PATH}" \
    --policy.type=pi0 \
    --policy.path=lerobot/pi0_base \
    --policy.dtype=bfloat16 \
    --training.steps="${MAX_STEPS}" \
    --training.batch_size="${BATCH_SIZE}" \
    --training.lr="${LEARNING_RATE}" \
    --training.grad_clip_norm=1.0 \
    --training.save_interval="${SAVE_INTERVAL}" \
    --training.output_dir="${CHECKPOINT_PATH}" \
    --training.gradient_checkpointing=true

echo "[done] Training complete. Checkpoints at: ${CHECKPOINT_PATH}"
ls -la "${CHECKPOINT_PATH}/" 2>/dev/null || true
