#!/bin/bash
#SBATCH --job-name=qwen3-8b-h200
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --exclusive
#SBATCH --partition=p5en
#SBATCH --time=24:00:00
#SBATCH --output=/fsx/ubuntu/qwen3-8b-pretraining/logs/%j.out
#SBATCH --error=/fsx/ubuntu/qwen3-8b-pretraining/logs/%j.err
#SBATCH --container-image=/fsx/ubuntu/qwen3-8b-pretraining/containers/nemo-efa-25.07.sqsh
#SBATCH --container-mounts=/fsx:/fsx

# Resolve head node BEFORE srun (scontrol not available inside container)
export MASTER_ADDR=$(/opt/slurm/bin/scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

# EFA / NCCL environment
export FI_PROVIDER=efa
export NCCL_SOCKET_IFNAME=^docker,lo,veth
export NCCL_DEBUG=WARN
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-tuner-aws-ofi.so
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/nccl/build/lib:$LD_LIBRARY_PATH

# Disable torch.compile (incompatible with this stack)
export TORCH_COMPILE_DISABLE=1

# Launch training
srun --container-env=MASTER_ADDR,MASTER_PORT,FI_PROVIDER,NCCL_SOCKET_IFNAME,NCCL_DEBUG,NCCL_TUNER_PLUGIN,LD_LIBRARY_PATH,TORCH_COMPILE_DISABLE \
    torchrun \
    --nnodes=${SLURM_NNODES} \
    --nproc-per-node=8 \
    --rdzv-id=${SLURM_JOB_ID} \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    /fsx/ubuntu/qwen3-8b-pretraining/code/train.py
