#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Stage 1: GPU sanity check via Pyxis/Enroot.
# Runs nvidia-smi inside a public CUDA container on one GPU node (8x H100).

#SBATCH --job-name=nvidia-smi
#SBATCH --partition=gpu-p5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=01-nvidia-smi_%j.out
#SBATCH --error=01-nvidia-smi_%j.out

set -euxo pipefail

srun --container-image=docker://nvidia/cuda:12.4.1-base-ubuntu22.04 \
     nvidia-smi
