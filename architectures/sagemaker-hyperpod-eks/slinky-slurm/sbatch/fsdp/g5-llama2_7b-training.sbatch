#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --nodes=4 # number of nodes to use
#SBATCH --job-name=llama2_7b-FSDP  # name of your job
#SBATCH --output=logs/%x_%j.out    # logfile for stdout
#SBATCH --error=logs/%x_%j.err     # logfile for stderr, remove it to merge both outputs
#SBATCH --exclusive                # job has exclusive use of the resource, no sharing
#SBATCH --ntasks-per-node=1        # one task per node
#SBATCH --cpus-per-task=32         # match the number of CPUs per node
set -ex;

###########################
###### User Variables #####
###########################

GPUS_PER_NODE=1

###########################
## Environment Variables ##
###########################

export CUDA_HOME="/usr/local/cuda"
export EFA_PATH="/opt/amazon/efa"
export OPEN_MPI_PATH="/opt/amazon/openmpi"
export OFI_NCCL_PATH="/opt/amazon/ofi-nccl"
export LD_LIBRARY_PATH="lib:${EFA_PATH}/lib:${OPEN_MPI_PATH}/lib:${CUDA_HOME}/lib64:/usr/local/lib:/lib/x86_64-linux-gnu:/opt/nccl/build/lib:${OFI_NCCL_PATH}/lib/x86_64-linux-gnu:/usr/local/nvidia/lib"

# LD_PRELOAD is required for PyTorch to find the NCCL library
export LD_PRELOAD="/usr/local/lib/libnccl.so.2"

export CUDA_VISIBLE_DEVICES=0     # Restrict PyTorch to only use the first GPU (GPU 0)
export NVIDIA_VISIBLE_DEVICES=all # Make all GPUs visible to NVIDIA container runtime

# Debug settings
export NCCL_DEBUG=INFO       # Set NCCL debug level for troubleshooting
export NCCL_DEBUG_SUBSYS=ALL # Enable detailed debugging output for all NCCL subsystems

# Timeout settings
export NCCL_TIMEOUT=1800           # Set overall NCCL operation timeout to 30 minutes (in seconds)
export NCCL_SOCKET_TIMEOUT=300     # Allow 5 minutes for TCP socket connections between nodes
export NCCL_ASYNC_ERROR_HANDLING=1 # Enable asynchronous error handling for better fault tolerance

# Buffer settings
export NCCL_BUFFSIZE=2097152 # Set NCCL communication buffer size to 2MB for larger transfers

# TCP connection settings
export TORCH_DISTRIBUTED_DETAILED_LOGGING=1 # Enable verbose logging for PyTorch distributed operations
export GLOO_SOCKET_IFNAME=eth0              # Use eth0 network interface for Gloo collective operations
export TP_SOCKET_IFNAME=eth0                # Use eth0 for tensor parallelism communication
export NCCL_SOCKET_IFNAME=eth0              # Use eth0 (primary EC2 network interface) for NCCL communication

# TCP Store timeout settings
export TORCHELASTIC_MAX_CALLTIME=3600 # Set maximum call time for TorchElastic operations to 1 hour
export PYTORCH_TIMEOUT=3600           # Set PyTorch RPC timeout to 1 hour
export TORCH_DISTRIBUTED_TIMEOUT=3600 # Set PyTorch distributed timeout to 1 hour

# PyTorch specific settings
export TORCH_DISTRIBUTED_DEBUG=DETAIL # Enable detailed debugging for distributed operations
export TORCH_CPP_LOG_LEVEL=INFO       # Set C++ frontend logging level to INFO
export CUDA_LAUNCH_BLOCKING=0         # Allow asynchronous CUDA kernel launches (0=async, 1=sync)

# HuggingFace settings
export HF_HUB_ETAG_TIMEOUT=60     # Metadata timeout (in seconds) for large clusters
export HF_TOKEN="<your-token-here>" # Token used to avoid throttling for data streaming

###########################
####### Torch Dist  #######
###########################

# Debug Slurm environment
echo "=== Slurm Environment ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "======================="

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=$GPUS_PER_NODE
    --nnodes=$SLURM_JOB_NUM_NODES
    --rdzv_id=$SLURM_JOB_ID
    --rdzv_backend=c10d
    --rdzv_endpoint=$(hostname)
)

export PATH="/usr/local/bin:$PATH"
export TRAIN_SCRIPT="/fsx/awsome-distributed-ai/examples/training/fsdp/src/train.py"
export PYTHONPATH="/usr/local/lib/python3.12/site-packages:$PYTHONPATH"
export TORCHRUN="/usr/local/bin/python3 -m torch.distributed.run"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

############################
# llama2_7b Training Params ##
############################
declare -a TRAINING_ARGS=(
    --max_context_width=512
    --num_key_value_heads=8
    --intermediate_size=2048
    --hidden_width=1024
    --num_layers=8
    --num_heads=16
    --model_type=llama_v2
    --tokenizer="hf-internal-testing/llama-tokenizer"
    --checkpoint_freq=100
    --validation_freq=100
    --max_steps=1000
    --checkpoint_dir=./checkpoints
    --dataset='allenai/c4'
    --dataset_config_name='en'
    --resume_from_checkpoint=./checkpoints
    --train_batch_size=1
    --val_batch_size=1
    --gradient_checkpointing=True
    --mixed_precision=bf16
    --sharding_strategy="full" # https://pytorch.org/docs/stable/fsdp.html
    --offload_activations=0
)

srun --export=ALL -l ${TORCHRUN} "${TORCHRUN_ARGS[@]}" $TRAIN_SCRIPT "${TRAINING_ARGS[@]}"