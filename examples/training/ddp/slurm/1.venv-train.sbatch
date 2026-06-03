#!/bin/bash
#SBATCH --job-name=ddp-venv
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1
#SBATCH --nodes 2
#SBATCH --output=logs/%x_%j.out # logfile for stdout/stderr

export LOGLEVEL=INFO

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=1     # For GPU: Set this to number of GPUs per node
    --nnodes=$SLURM_JOB_NUM_NODES
    --rdzv_id=$SLURM_JOB_ID
    --rdzv_backend=c10d
    --rdzv_endpoint=$(hostname)
)

declare -a TRAIN_ARGS=(
    --total_epochs 500
    --save_every 1
    --batch_size 32
    --checkpoint_path ./snapshot.pt
    --use_mlflow
)

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi

srun ${AUTO_RESUME} ./pt/bin/torchrun \
    "${TORCHRUN_ARGS[@]}" \
    $(dirname "$PWD")/ddp.py ${TRAIN_ARGS[@]}
