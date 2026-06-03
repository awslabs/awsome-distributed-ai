#!/bin/bash
#SBATCH --nodes=4                    # number of nodes
#SBATCH --ntasks-per-node=8          # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=8
#SBATCH --exclusive                   # exclusive node access
#SBATCH --output slurm-esm1nv-train-%j.out

export FI_EFA_USE_HUGE_PAGE=0


###########################
###### User Variables #####
###########################

# default variables for Enroot
: "${IMAGE:=$(pwd)/${ENROOT_IMAGE}}"
: "${DATA_PATH:=/fsx}"
: "${FSX_MOUNT:=$DATA_PATH:$DATA_PATH}"

declare -a ARGS=(
    --container-image $IMAGE
    --container-mount-home
    --container-mounts $FSX_MOUNT
)


# Training parameters
# =========================
MICRO_BATCH_SIZE=256 # micro batch size per GPU, for best efficiency should be set to occupy ~85% of GPU memory. Suggested value for A100 80GB is 256
ACCUMULATE_GRAD_BATCHES=1 # gradient accumulation
TENSOR_MODEL_PARALLEL_SIZE=1 # tensor model parallel size
VAL_CHECK_INTERVAL=500 # how often validation step is performed, including downstream task validation
MAX_STEPS=1000000 # duration of training as the number of training steps
# =========================


# Logging
# =========================
PROJECT_NAME="esm1nv_pretraining" # project name, will be used for logging
EXP_TAG="-small" # any additional experiment info, can be empty
EXP_NAME="esm1nv_batch${MICRO_BATCH_SIZE}_gradacc${ACCUMULATE_GRAD_BATCHES}_nodes${SLURM_JOB_NUM_NODES}${EXP_TAG}"
CREATE_WANDB_LOGGER=False # set to False if you don't want to log results with WandB
WANDB_LOGGER_OFFLINE=False # set to True if there are issues uploading to WandB during training
# =========================

# Mounts
# =========================
DATA_PATH=/fsx/processed # Directory with data for model training and downstream task validation
TRAIN_FILES='x_OP_000..049_CL_' # Range for the train dataset
TEST_FILES='x_OP_000..049_CL_'  # Range for the test dataset
VAL_FILES='x_OP_000..049_CL_'   # Range for the val dataset
RESULTS_PATH=/fsx/esm1nv-train/${PROJECT_NAME}/${EXP_NAME}/results # directory to store logs, checkpoints and results

mkdir -p ${RESULTS_PATH}}


# Necessary Exports
# =========================
export HYDRA_FULL_ERROR=1
# =========================

srun -l "${ARGS[@]}"  python3 /workspace/bionemo/examples/protein/esm1nv/pretrain.py \
    --config-path=/workspace/bionemo/examples/protein/esm1nv/conf \
    --config-name=pretrain_small \
    exp_manager.exp_dir=${RESULTS_PATH} \
    exp_manager.create_wandb_logger=${CREATE_WANDB_LOGGER} \
    exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME} \
    ++exp_manager.wandb_logger_kwargs.offline=${WANDB_LOGGER_OFFLINE} \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    trainer.devices=${SLURM_GPUS_PER_NODE} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES} \
    trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
    model.micro_batch_size=${MICRO_BATCH_SIZE} \
    model.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
    model.data.dataset_path=${DATA_PATH} \
    model.data.dataset.train=${TRAIN_FILES} \
    model.data.dataset.val=${VAL_FILES} \
    model.data.dataset.test=${TEST_FILES} \
    model.data.index_mapping_dir=${DATA_PATH} \
    ++model.dwnstr_task_validation.enabled=False
