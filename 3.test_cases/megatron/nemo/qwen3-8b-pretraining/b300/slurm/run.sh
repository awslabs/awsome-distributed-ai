#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#SBATCH --job-name=qwen3-8b-b300
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --exclusive
#SBATCH --output=/fsx/ubuntu/qwen3-8b-pretraining/logs/%j.out
#SBATCH --error=/fsx/ubuntu/qwen3-8b-pretraining/logs/%j.err

# EFA / NCCL environment
export FI_PROVIDER=efa
export NCCL_SOCKET_IFNAME=^docker,lo,veth
export NCCL_DEBUG=WARN
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-tuner-aws-ofi.so
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Launch - Megatron uses SLURM env vars (SLURM_PROCID, SLURM_LOCALID) for distributed init
/opt/slurm/bin/srun --mpi=pmix --container-image=/fsx/ubuntu/qwen3-8b-pretraining/containers/nemo-efa-26.04.sqsh --container-mounts=/fsx:/fsx,/opt/slurm:/opt/slurm --container-env=FI_PROVIDER,NCCL_SOCKET_IFNAME,NCCL_DEBUG,NCCL_TUNER_PLUGIN,LD_LIBRARY_PATH,TORCH_COMPILE_DISABLE python /fsx/ubuntu/awsome-distributed-ai/3.test_cases/megatron/nemo/qwen3-8b-pretraining/h200/train.py
