#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Download Something-Something v2 dataset for V-JEPA 2 training.
#
# IMPORTANT: SSv2 requires you to accept terms of use before downloading.
# Visit https://developer.qualcomm.com/software/ai-datasets/something-something
# to check if you need to request access.
#
# This script assumes you have already downloaded the SSv2 videos and
# annotation files to the DATASET_DIR below.
#
# If downloading from HuggingFace (requires HF_TOKEN):
#   pip install datasets
#   python -c "from datasets import load_dataset; ds = load_dataset('HuggingFaceM4/something_something_v2')"

#SBATCH --job-name=vjepa2-download-ssv2
#SBATCH --output=logs/download_ssv2/%A.out
#SBATCH --error=logs/download_ssv2/%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=p5en

set -euo pipefail

VJEPA2_DIR=/fsx/${USER}/vjepa2
DATASET_DIR=${VJEPA2_DIR}/datasets/ssv2
mkdir -p ${DATASET_DIR}/videos
mkdir -p logs/download_ssv2

###########################
## Container Variables ####
###########################

export CONTAINER_IMAGE=${VJEPA2_DIR}/vjepa2.sqsh
export FSX_MOUNT=/fsx:/fsx

declare -a ARGS=(
    --container-image $CONTAINER_IMAGE
    --container-mounts $FSX_MOUNT
)

###########################
## Download & Prepare #####
###########################

# Option 1: If SSv2 videos are already downloaded, just generate the CSV
# Assumes: videos are in ${DATASET_DIR}/videos/
#          labels.json is in ${DATASET_DIR}/labels.json
#          train.json is in ${DATASET_DIR}/train.json

srun "${ARGS[@]}" python /vjepa2/scripts/prepare_ssv2.py \
    --video_dir ${DATASET_DIR}/videos \
    --labels_json ${DATASET_DIR}/labels.json \
    --train_json ${DATASET_DIR}/train.json \
    --output_csv ${DATASET_DIR}/ssv2_train_paths.csv

echo "Dataset preparation complete."
echo "CSV written to: ${DATASET_DIR}/ssv2_train_paths.csv"
