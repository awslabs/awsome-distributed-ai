#!/bin/bash

#SBATCH -N 1 # number of nodes to use
#SBATCH --job-name=c4-preprocess # name of your job
#SBATCH --output=logs/%x_%j.out # logfile for stdout
#SBATCH --error=logs/%x_%j.err # logfile for stderr, remove it to merge both outputs
#SBATCH --ntasks-per-node 8 # Number of processes per node
#SBATCH --exclusive

# default variables for Enroot, if these variables are defined then use them
: "${APPS_PATH:=/apps}"
: "${IMAGE:=$APPS_PATH/llm-foundry.sqsh}"
: "${FSX_PATH:=/fsx}"
: "${DATA_PATH:=$FSX_PATH/c4-dataset}"
: "${CONTAINER_MOUNT:=$FSX_PATH:$FSX_PATH}"

declare -a ARGS=(
    --container-image $IMAGE
    --container-mounts $CONTAINER_MOUNT
)

srun -l "${ARGS[@]}" python /llm-foundry/scripts/data_prep/convert_dataset_hf.py \
  --dataset c4 --data_subset en \
  --out_root ${DATA_PATH} --splits train_small val_small \
  --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>'
