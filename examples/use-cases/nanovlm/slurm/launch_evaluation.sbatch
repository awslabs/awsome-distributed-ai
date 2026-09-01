#!/bin/bash
#SBATCH --job-name=nanovlm_eval_workshop
#SBATCH --output=logs/eval/%A_%a.out
#SBATCH --error=logs/eval/%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --partition=p5en
#SBATCH --array=0

cd ..

mkdir -p logs/eval

export CONTAINER_IMAGE=$(pwd)/nanovlm.sqsh

export FSX_MOUNT=$(pwd):$(pwd)
export PYTHONPATH=$(pwd)/nanoVLM:$PYTHONPATH

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

declare -a ARGS=(
    --container-image $CONTAINER_IMAGE
    --container-mounts $FSX_MOUNT
)

export EVAL_SCRIPT=utils/run_checkpoint_evaluations.py

declare -a EVAL_ARGS=(
    --checkpoints_dir $CHECKPOINT_DIR
    --steps 450
    --eval_tasks mmstar,mmmu 
    --batch_size 1
)

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi

srun ${AUTO_RESUME} -l "${ARGS[@]}" python $EVAL_SCRIPT "${EVAL_ARGS[@]}"
