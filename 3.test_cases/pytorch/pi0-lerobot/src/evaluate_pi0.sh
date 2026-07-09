#!/bin/bash
# Pi0 (LeRobot) — Evaluation Entrypoint for HyperPod EKS
#
# Runs evaluate_pi0.py against a fine-tuned checkpoint on FSx.
# Expects:
#   DATASET          - 'droid' or 'libero'
#   CHECKPOINT_PATH  - Path to pretrained_model/ directory on FSx
#   TEST_DATA_PATH   - Path to the test dataset on FSx
#   RESULTS_PATH     - Where to write results JSON

set -euo pipefail

echo "============================================"
echo " Pi0 (LeRobot) — HyperPod EKS Evaluation"
echo "============================================"

DATASET="${DATASET:-droid}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/data/runs/pi0-droid/checkpoints/002000/pretrained_model}"
TEST_DATA_PATH="${TEST_DATA_PATH:-/data/datasets/${DATASET}_test}"
RESULTS_PATH="${RESULTS_PATH:-/data/runs/pi0-${DATASET}/eval_results.json}"

echo "  Dataset:    ${DATASET}"
echo "  Checkpoint: ${CHECKPOINT_PATH}"
echo "  Test data:  ${TEST_DATA_PATH}"
echo "  Results:    ${RESULTS_PATH}"

# HuggingFace login for base model download
if [ -n "${HF_TOKEN:-}" ]; then
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" 2>/dev/null || true
fi

# Apply local dataset patch
export HF_HUB_OFFLINE=1
python /opt/pi0-lerobot/src/lerobot_local_patch.py 2>/dev/null || true

# Run evaluation
python /opt/pi0-lerobot/src/evaluate_pi0.py \
    --dataset "${DATASET}" \
    --finetuned-path "${CHECKPOINT_PATH}" \
    --test-dataset-local "${TEST_DATA_PATH}" \
    --results-out "${RESULTS_PATH}" \
    --num-trajectories 5 \
    --num-inference-steps 10 5 3 1

echo "[done] Results saved to: ${RESULTS_PATH}"
