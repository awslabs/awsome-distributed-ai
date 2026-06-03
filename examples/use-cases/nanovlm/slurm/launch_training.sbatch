#!/bin/bash
#SBATCH --job-name=train_nanoVLM
#SBATCH --output=logs/train_nanoVLM/%A.out
#SBATCH --error=logs/train_nanoVLM/%A.err
#SBATCH --time=01:00:00
#SBATCH --nodes=4
#SBATCH --partition=p5en

GPUS_PER_NODE=8 #set to 1 for g5.8xlarge

cd ..

export CONTAINER_IMAGE=$(pwd)/nanovlm.sqsh

export FSX_MOUNT=$(pwd):$(pwd)


export NCCL_DEBUG=INFO
export FI_PROVIDER=efa
#export FI_EFA_USE_HUGE_PAGE=0    # Set to 0 when you see os.fork() causes OSError: Cannot allocate memory.  Disabling huge page causes minor performance hit.
## Switching SYNC_MEMOPS to zero can boost throughput with FSDP
## Disables CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
## Reduces memory synchronizations
## https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
# LD_PRELOAD is required for PyTorch to find the NCCL library
# This path assumes you are using the Deep Learning AMI
# If you are not using the DLAMI, you may need to update this path
export LD_PRELOAD=/usr/local/cuda-12.8/lib/libnccl.so
export NCCL_SOCKET_IFNAME=^docker,lo,veth,eth

declare -a ARGS=(
    --container-image $CONTAINER_IMAGE
    --container-mounts $FSX_MOUNT
)

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=$GPUS_PER_NODE
    --nnodes=$SLURM_JOB_NUM_NODES
    --rdzv_id=$SLURM_JOB_ID
    --rdzv_backend=c10d
    --rdzv_endpoint=$SLURMD_NODENAME:29500 
)

export TRAIN_SCRIPT=train.py

declare -a TRAINING_ARGS=(
    --no_log_wandb
)

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi

srun ${AUTO_RESUME}  -l "${ARGS[@]}" torchrun "${TORCHRUN_ARGS[@]}" $TRAIN_SCRIPT "${TRAINING_ARGS[@]}"