#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --exclusive
#SBATCH --job-name=deepspeed-pretrain-103b
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euxo pipefail

# ============================================================
# Environment defaults
# ============================================================
: "${APPS_PATH:=/fsx/apps}"
: "${IMAGE:=$APPS_PATH/deepspeed.sqsh}"
: "${FSX_PATH:=/fsx}"
: "${DATA_DIR:=$FSX_PATH/deepspeed/data}"
: "${MEGATRON_DS_PATH:=$FSX_PATH/deepspeed/Megatron-DeepSpeed}"

# ============================================================
# Parallelism config (overridable via env vars from sweep_runner.sh)
# ============================================================
: "${TP:=8}"
: "${PP:=2}"
: "${ZERO_STAGE:=1}"
: "${MICRO_BATCH_SIZE:=1}"
: "${GLOBAL_BATCH_SIZE:=64}"
: "${TRAIN_ITERS:=50}"

# ============================================================
# ~103B GPT model architecture
# Layers=80, Hidden=12288, Heads=96, FFN=49152
# Estimated parameters: ~103B
# ============================================================
: "${NUM_LAYERS:=80}"
: "${HIDDEN_SIZE:=12288}"
: "${NUM_HEADS:=96}"
: "${FFN_HIDDEN_SIZE:=49152}"
: "${SEQ_LENGTH:=2048}"

# ============================================================
# Optional features (set to 1 to enable)
# ============================================================
: "${USE_ACTIVATION_CHECKPOINTING:=0}"
: "${USE_SEQUENCE_PARALLEL:=0}"
: "${USE_OVERLAP_COMM:=0}"
: "${ENABLE_FUSIONS:=0}"
: "${CONFIG_NAME:=baseline}"

# ============================================================
# PyTorch memory allocator optimisation
# ============================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# Cluster topology
# ============================================================
export NODES=( $( scontrol show hostnames "$SLURM_JOB_NODELIST" ) )
export NODES_ARRAY=($NODES)
export HEAD_NODE=${NODES_ARRAY[0]}
export MASTER_ADDR=$(hostname --ip-address)
export MASTER_PORT=$((RANDOM + 10000))
export NNODES=$SLURM_JOB_NUM_NODES
export NUM_GPUS_PER_NODE=8

# ============================================================
# Network settings for EFA + NCCL
# ============================================================
export FI_LOG_LEVEL=1
export FI_PROVIDER=efa
export FI_EFA_USE_HUGE_PAGE=0
export NCCL_SOCKET_IFNAME=^docker,lo,veth
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_BUFFERSIZE=8388608
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so
export NCCL_ASYNC_ERROR_HANDLING=1
export OMPI_MCA_plm=^slurm

# ============================================================
# Generate DeepSpeed config dynamically
# ============================================================
mkdir -p configs logs

PRESCALE_GRAD="false"
if [ "${ZERO_STAGE}" -eq 0 ]; then
    PRESCALE_GRAD="true"
fi

OVERLAP_COMM_BOOL="false"
if [ "${USE_OVERLAP_COMM}" -eq 1 ]; then
    OVERLAP_COMM_BOOL="true"
fi

# Build ZeRO optimisation block depending on stage
if [ "${ZERO_STAGE}" -eq 3 ]; then
    ZERO_BLOCK=$(cat <<ZEOF
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": true,
        "allgather_bucket_size": 500000000,
        "overlap_comm": ${OVERLAP_COMM_BOOL},
        "reduce_scatter": true,
        "reduce_bucket_size": 500000000,
        "contiguous_gradients": true,
        "stage3_prefetch_bucket_size": 500000000,
        "stage3_param_persistence_threshold": 1000000,
        "stage3_max_live_parameters": 1000000000,
        "stage3_max_reuse_distance": 1000000000
    }
ZEOF
    )
else
    ZERO_BLOCK=$(cat <<ZEOF
    "zero_optimization": {
        "stage": ${ZERO_STAGE},
        "allgather_partitions": true,
        "allgather_bucket_size": 500000000,
        "overlap_comm": ${OVERLAP_COMM_BOOL},
        "reduce_scatter": true,
        "reduce_bucket_size": 500000000,
        "contiguous_gradients": true
    }
ZEOF
    )
fi

cat <<EOF > configs/ds_config_run.json
{
    "train_batch_size": ${GLOBAL_BATCH_SIZE},
    "train_micro_batch_size_per_gpu": ${MICRO_BATCH_SIZE},
    "steps_per_print": 10,
${ZERO_BLOCK},
    "gradient_clipping": 1.0,
    "prescale_gradients": ${PRESCALE_GRAD},
    "bf16": {
        "enabled": true
    },
    "wall_clock_breakdown": false
}
EOF

# ============================================================
# Hostfile for DeepSpeed
# ============================================================
export HOSTFILE="/fsx/hostfile_${SLURM_JOB_ID}"
function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=8 if $slots==0;
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > "${HOSTFILE}"

# ============================================================
# Container + distributed launch args
# ============================================================
declare -a SRUN_ARGS=(
    --container-image "${IMAGE}"
    --container-mounts /fsx,/opt/slurm/bin
)

declare -a DIST_ARGS=(
    --nnodes ${NNODES}
    --nproc-per-node ${NUM_GPUS_PER_NODE}
    --master_addr ${MASTER_ADDR}
    --master_port ${MASTER_PORT}
    --rdzv_id $RANDOM
    --rdzv_backend c10d
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT}
)

# ============================================================
# Model + training args
# ============================================================
declare -a MODEL_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --num-attention-heads ${NUM_HEADS}
    --ffn-hidden-size ${FFN_HIDDEN_SIZE}
    --seq-length ${SEQ_LENGTH}
    --max-position-embeddings ${SEQ_LENGTH}
    --micro-batch-size ${MICRO_BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --train-iters ${TRAIN_ITERS}
    --lr 1.0e-4
    --min-lr 1.0e-6
    --lr-decay-style cosine
    --lr-warmup-iters 5
    --lr-decay-iters 50
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --log-interval 1
    --eval-iters 0
    --eval-interval 1000
    --bf16
    --data-path ${DATA_DIR}/BookCorpusDataset_text_document
    --vocab-file ${DATA_DIR}/gpt2-vocab.json
    --merge-file ${DATA_DIR}/gpt2-merges.txt
    --split 100,0,0
    --data-impl mmap
    --num-workers 0
)

# By default disable fusions (matches sweep v1 behaviour).
# Set ENABLE_FUSIONS=1 to enable them.
if [ "${ENABLE_FUSIONS}" -eq 0 ]; then
    MODEL_ARGS+=(
        --no-masked-softmax-fusion
        --no-bias-gelu-fusion
        --no-bias-dropout-fusion
        --no-gradient-accumulation-fusion
    )
fi

declare -a DS_ARGS=(
    --tensor-model-parallel-size ${TP}
    --zero-stage ${ZERO_STAGE}
    --deepspeed_config ${PWD}/configs/ds_config_run.json
    --deepspeed
    --distributed-backend nccl
)

# Megatron-DeepSpeed wraps the model in PipelineModule by default.
# DeepSpeed's PipelineEngine asserts that ZeRO stage < 2.
# When using ZeRO-2 or ZeRO-3 we must disable pipeline parallel entirely
# via --no-pipeline-parallel (which sets ds_pipeline_enabled=False).
if [ "${ZERO_STAGE}" -ge 2 ]; then
    DS_ARGS+=(
        --pipeline-model-parallel-size 1
        --no-pipeline-parallel
    )
else
    DS_ARGS+=(--pipeline-model-parallel-size ${PP})
fi

# ============================================================
# Optional features
# ============================================================
if [ "${USE_ACTIVATION_CHECKPOINTING}" -eq 1 ]; then
    DS_ARGS+=(
        --checkpoint-activations
        --deepspeed-activation-checkpointing
    )
fi

if [ "${USE_SEQUENCE_PARALLEL}" -eq 1 ]; then
    DS_ARGS+=(--sequence-parallel)
fi

# ============================================================
# Launch training
# Note: Using python3 -m torch.distributed.run instead of torchrun
# because the container's Python version may differ from the host
# ============================================================
echo "=== DeepSpeed 103B GPT Pretraining ==="
echo "Config: ${CONFIG_NAME}"
echo "Nodes: ${NNODES}, GPUs/node: ${NUM_GPUS_PER_NODE}, Total GPUs: $((NNODES * NUM_GPUS_PER_NODE))"
echo "TP=${TP}, PP=${PP}, ZeRO=${ZERO_STAGE}"
echo "MBS=${MICRO_BATCH_SIZE}, GBS=${GLOBAL_BATCH_SIZE}"
echo "Model: layers=${NUM_LAYERS}, hidden=${HIDDEN_SIZE}, heads=${NUM_HEADS}, ffn=${FFN_HIDDEN_SIZE}"
echo "Seq length: ${SEQ_LENGTH}, Fusions: ${ENABLE_FUSIONS}"
echo "Activation ckpt: ${USE_ACTIVATION_CHECKPOINTING}, Seq parallel: ${USE_SEQUENCE_PARALLEL}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-}"
echo "======================================="

# Convert arrays to strings for bash -c invocation
DIST_ARGS_STR="${DIST_ARGS[*]}"
MODEL_ARGS_STR="${MODEL_ARGS[*]}"
DS_ARGS_STR="${DS_ARGS[*]}"

# Note: Variables inside the bash -c string are expanded on the host side before
# being passed to the container. This is intentional — the container does not have
# access to these env vars at shell expansion time.
srun -l "${SRUN_ARGS[@]}" bash -c "export PYTHONPATH=${MEGATRON_DS_PATH} && cd ${MEGATRON_DS_PATH} && python3 -m torch.distributed.run ${DIST_ARGS_STR} pretrain_gpt.py ${MODEL_ARGS_STR} ${DS_ARGS_STR}"

# Cleanup hostfile
rm -f "${HOSTFILE}"
