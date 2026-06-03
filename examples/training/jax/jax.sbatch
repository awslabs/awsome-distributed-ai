#!/bin/bash

#SBATCH -o jax_%j.out
#SBATCH -e jax_%j.err
#SBATCH -n 384
#SBATCH --gpus-per-node=8
#SBATCH --exclusive

GPU_PER_NODE=8
TOTAL_NB_GPUS=$(($SLURM_JOB_NUM_NODES * $GPU_PER_NODE))

# Shared file system and container directories
export SHARED_FS_DIR=/fsx/data
export CONTAINER_DIR=/data

# EFA Flags
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1

# NCCL Flags
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Library Path
export LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/aws-ofi-nccl/install/lib:/usr/local/cuda-12/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# XLA Configuration
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_simplify_all_fp_conversions --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true --xla_gpu_enable_highest_priority_async_stream=true --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_all_reduce_combine_threshold_bytes=33554432 --xla_gpu_graph_level=0 --xla_gpu_enable_async_all_reduce=true"
export TPU_TYPE=gpu
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Setup and results directory
export LEAD_NODE=${SLURMD_NODENAME} 
export BASE_DIR=${CONTAINER_DIR}/700/$SLURM_JOBID

# Create results directory on shared file system
CHECKPOINT_DIR=/fsx/${BASE_DIR}
mkdir -p ${CHECKPOINT_DIR}/checkpoints
mkdir -p ${CHECKPOINT_DIR}/LOG_DIR


# JAX Configuration
export TRAINING_CONFIG=paxml.tasks.lm.params.lm_cloud.LmCloudSpmd2BLimitSteps
export JAX_FLAGS="--fdl.ICI_MESH_SHAPE=[1,${TOTAL_NB_GPUS},1] --fdl.PERCORE_BATCH_SIZE=32"

srun --container-image /fsx/paxml_jax-0.4.18-1.2.0.sqsh --container-mounts ${SHARED_FS_DIR}:${CONTAINER_DIR} -n ${TOTAL_NB_GPUS} -N ${SLURM_JOB_NUM_NODES} /bin/bash run_paxml.sh