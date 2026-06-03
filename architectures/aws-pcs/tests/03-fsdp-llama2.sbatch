#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Stage 3: multi-node FSDP training smoke test (PyTorch FSDP, Llama-2 7B).
# Runs the repo's FSDP test case (3.test_cases/pytorch/FSDP) on 2 GPU nodes.
#
# Prereqs (run once on the login node, see tests/README.md):
#   - venv at /fsx/fsdp-env with the FSDP requirements installed
#   - repo cloned to /fsx/awsome-distributed-ai
#   - HF_TOKEN exported (gated tokenizer); HF_HOME on /fsx (Lustre), NOT /home (NFS)
#
# Adjust --partition to your GPU queue (e.g. gpu-p5 / gpu-p6b200 / gpu-p6b300).

#SBATCH --job-name=fsdp-llama2
#SBATCH --partition=gpu-p6b200
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1        # ONE torchrun per node; it spawns the 8 GPU procs
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=03-fsdp-llama2_%j.out
#SBATCH --error=03-fsdp-llama2_%j.err

set -ex
GPUS_PER_NODE=8

# EFA / NCCL. Let NCCL auto-select the bootstrap interface — do NOT pin a single
# NIC on the multi-NIC P5/P6 (all interfaces share one subnet; pinning one breaks
# the cross-node bootstrap ring).
export FI_PROVIDER=efa
export NCCL_SOCKET_IFNAME=^docker,lo
export FI_EFA_USE_HUGE_PAGE=0
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}
export NCCL_DEBUG=INFO

# HuggingFace: cache on Lustre (/fsx), never the NFS /home (stale-handle errors).
export HF_HOME=/fsx/.hf-cache
export HF_HUB_ETAG_TIMEOUT=60
# export HF_TOKEN=hf_xxx   # set in your environment / sbatch --export

source /fsx/fsdp-env/bin/activate
cd /fsx/awsome-distributed-ai/3.test_cases/pytorch/FSDP/slurm

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=${GPUS_PER_NODE}
    --nnodes=${SLURM_JOB_NUM_NODES}
    --rdzv_id=${SLURM_JOB_ID}
    --rdzv_backend=c10d
    --rdzv_endpoint=$(hostname)
)

declare -a TRAINING_ARGS=(
    --max_context_width=4096 --num_key_value_heads=32 --intermediate_size=11008
    --hidden_width=4096 --num_layers=32 --num_heads=32 --model_type=llama_v2
    --tokenizer=hf-internal-testing/llama-tokenizer --checkpoint_freq=50
    --validation_freq=100 --max_steps=100 --checkpoint_dir=./checkpoints
    --dataset=allenai/c4 --dataset_config_name=en --resume_from_checkpoint=./checkpoints
    --train_batch_size=1 --val_batch_size=1 --sharding_strategy=full --offload_activations=1
)

srun -l torchrun "${TORCHRUN_ARGS[@]}" ../src/train.py "${TRAINING_ARGS[@]}"
