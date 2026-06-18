#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#SBATCH --job-name=qwen3-8b-b300
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --exclusive
#SBATCH --partition=b300
#SBATCH --time=24:00:00
#SBATCH --output=/fsx/ubuntu/qwen3-8b/logs/%j.out
#SBATCH --error=/fsx/ubuntu/qwen3-8b/logs/%j.err
#SBATCH --container-image=/fsx/ubuntu/qwen3-8b/containers/nemo-efa-26.02.sqsh
#SBATCH --container-mounts=/fsx:/fsx

# EFA / NCCL environment
export FI_PROVIDER=efa
export NCCL_SOCKET_IFNAME=^docker,lo,veth
export NCCL_DEBUG=WARN
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-tuner-aws-ofi.so
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}
export TORCH_COMPILE_DISABLE=1

# Copy training script to expected location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p /fsx/ubuntu/qwen3-8b/code
cp "${SCRIPT_DIR}/train.py" /fsx/ubuntu/qwen3-8b/code/train.py

# Launch - Megatron uses SLURM env vars (SLURM_PROCID, SLURM_LOCALID) for distributed init
srun --container-env=FI_PROVIDER,NCCL_SOCKET_IFNAME,NCCL_DEBUG,NCCL_TUNER_PLUGIN,LD_LIBRARY_PATH,TORCH_COMPILE_DISABLE \
    python /fsx/ubuntu/qwen3-8b/code/train.py
