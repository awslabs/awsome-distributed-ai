#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#SBATCH --job-name=preprocess-c4
#SBATCH --partition=p5en
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/fsx/ubuntu/qwen3-8b-pretraining/logs/preprocess-%j.out
#SBATCH --export=ALL

# --- HF_TOKEN must be set before submitting ---
# export HF_TOKEN=<your token>
# sbatch preprocess.sh
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set. Export it before submitting:"
    echo "  export HF_TOKEN=<your token> && sbatch preprocess.sh"
    exit 1
fi

export HF_HOME="/fsx/ubuntu/.cache/huggingface"

mkdir -p /fsx/ubuntu/qwen3-8b-pretraining/logs
mkdir -p /fsx/ubuntu/qwen3-8b-pretraining/datasets

# Create a virtual environment for the preprocessing
python3 -m venv /fsx/ubuntu/qwen3-8b-pretraining/venv
source /fsx/ubuntu/qwen3-8b-pretraining/venv/bin/activate

PYTHON=/fsx/ubuntu/qwen3-8b-pretraining/venv/bin/python
SCRIPT_DIR=$SLURM_SUBMIT_DIR/preprocessing/
SCRIPT="${SCRIPT_DIR}/preprocess.py"

pip install -r $SCRIPT_DIR/requirements.txt

echo "=== C4 Preprocessing ==="
echo "Node: $(hostname) | CPUs: $(nproc) | Start: $(date)"

$PYTHON $SCRIPT \
    --output-prefix /fsx/ubuntu/qwen3-8b-pretraining/datasets/c4_qwen3_8b \
    --tokenizer Qwen/Qwen3-8B \
    --num-tokens 1000000000 \
    --workers $(nproc) \
    --cache-dir /fsx/ubuntu/qwen3-8b-pretraining/cache/c4

echo "Finished: $(date)"
