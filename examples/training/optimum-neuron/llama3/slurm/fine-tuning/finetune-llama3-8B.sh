#!/bin/bash

###########################
## Auto-detect Instance  ##
###########################

# Detect NeuronCore count and set parallelism accordingly.
# Based on upstream: https://github.com/huggingface/optimum-neuron/blob/main/examples/training/llama/finetune_llama.sh
NEURON_CORES=$(neuron-ls | awk '/^\| [0-9]+/ {total += $4} END {print total}')

if [ "$NEURON_CORES" -eq 32 ]; then
    # trn1.32xlarge / trn1n.32xlarge: 32 NeuronCores, TP=8
    PROCESSES_PER_NODE=32
    TP_DEGREE=8
elif [ "$NEURON_CORES" -eq 64 ]; then
    # trn2.48xlarge: 64 NeuronCores (LNC=2), TP=4
    PROCESSES_PER_NODE=64
    TP_DEGREE=4
elif [ "$NEURON_CORES" -eq 4 ]; then
    # trn2.3xlarge: 4 NeuronCores (LNC=2), TP=4
    PROCESSES_PER_NODE=4
    TP_DEGREE=4
else
    echo "ERROR: Unsupported NeuronCore count: $NEURON_CORES"
    echo "Supported instances: trn1.32xlarge (32), trn2.48xlarge (64), trn2.3xlarge (4)"
    exit 1
fi

echo "Detected $NEURON_CORES NeuronCores: PROCESSES_PER_NODE=$PROCESSES_PER_NODE, TP_DEGREE=$TP_DEGREE"

###########################
###### User Variables #####
###########################

if [ $NEURON_EXTRACT_GRAPHS_ONLY -gt 0 ]; then
    MAX_STEPS=10
    MAYBE_COMPILE="neuron_parallel_compile"
    OUTPUT_DIR="/fsx/ubuntu/peft_ft/compile"
else
    MAX_STEPS=-1
    OUTPUT_DIR="/fsx/ubuntu/peft_ft/model_checkpoints"
fi

###########################
## Environment Variables ##
###########################

CACHE_DIR='/fsx/ubuntu/peft_ft/cache/neuron_compile_cache/llama3-8B'
mkdir -p $CACHE_DIR
export NEURON_CC_FLAGS="--model-type=transformer --distribution-strategy=llm-training --enable-saturate-infinity --cache_dir=$CACHE_DIR"
export OMP_NUM_THREADS=1
export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=5
export NEURON_RT_STOCHASTIC_ROUNDING_EN=1
export MALLOC_ARENA_MAX=70
export FI_PROVIDER="efa"

###########################
####### Torch Dist  #######
###########################

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=$PROCESSES_PER_NODE
    --nnodes=$SLURM_JOB_NUM_NODES
)

export TRAIN_SCRIPT=/fsx/ubuntu/awsome-distributed-training/examples/pytorch/optimum-neuron/llama3/src/train.py

############################
##### Training Params ######
############################

# Script-specific arguments (ScriptArguments dataclass)
# NeuronTrainingArguments are passed as standard HuggingFace training args
declare -a TRAINING_ARGS=(
    --model_id "/fsx/ubuntu/peft_ft/model_artifacts/llama3-8B" \
    --dataset "databricks/databricks-dolly-15k" \
    --max_seq_length 2048 \
    --model_final_path "/fsx/ubuntu/peft_ft/model_checkpoints/final" \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 \
    --num_train_epochs 1 \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 3 \
    --learning_rate 2e-05 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --tensor_parallel_size $TP_DEGREE \
    --logging_steps 1 \
    --save_steps 400 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir
)

source /fsx/ubuntu/peft_ft/env_llama3_8B_peft/bin/activate

$MAYBE_COMPILE torchrun "${TORCHRUN_ARGS[@]}" $TRAIN_SCRIPT "${TRAINING_ARGS[@]}"
