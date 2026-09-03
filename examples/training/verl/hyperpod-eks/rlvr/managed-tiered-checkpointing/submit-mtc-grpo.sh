#!/usr/bin/env bash
set -xeuo pipefail

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../setup/env_vars"

# Project configuration
project_name='verl_grpo_example_gsm8k'
exp_name='qwen3_0.6b_function_rm'

# GRPO Algorithm parameters
adv_estimator=grpo
use_kl_in_reward=False
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl
entropy_coeff=0

# Token length configuration
max_prompt_length=512
max_response_length=1024
filter_overlong_prompts=True
truncation='error'

# Training configuration
train_prompt_bsz=${TRAIN_BATCH_SIZE:-32}  # Total batch size
gen_prompt_bsz=${GEN_BATCH_SIZE:-$train_prompt_bsz}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-5}
train_prompt_mini_bsz=32  # Must be <= train_batch_size
train_prompt_micro_bsz_per_gpu=1

# Ray configuration
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}

# Cluster configuration
NNODES=${NUM_NODES:-4}
GPUS_PER_NODE=${NUM_GPU_PER_NODE:-4}

# Model and data paths
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-0.6B"}
RAY_DATA_HOME=${RAY_DATA_HOME:-"/fsx/verl"}

# Data files - using GSM8K dataset
TRAIN_FILE="${RAY_DATA_HOME}/data/gsm8k/train.parquet"
TEST_FILE="${RAY_DATA_HOME}/data/gsm8k/test.parquet"

# S3 checkpoint configuration for Managed Tiered Checkpointing (MTC)
# MTC enables asynchronous checkpointing to CPU memory with automatic background sync to S3
# This reduces checkpoint overhead during training and provides faster recovery
# Learn more: https://docs.aws.amazon.com/sagemaker/latest/dg/managed-tier-checkpointing.html
S3_CHECKPOINT_BASE=${S3_CHECKPOINT_BASE:?Error: S3_CHECKPOINT_BASE not set. Please configure in env_vars file.}

# Performance parameters
gen_tp=2
log_prob_micro_bsz_per_gpu=32
gpu_memory_utilization=0.6

# Memory optimization
param_offload=False
optimizer_offload=False
ref_param_offload=True

# Print configuration for verification
echo "=== MTC GRPO Training Configuration ==="
echo "Project: ${project_name}"
echo "Experiment: ${exp_name}"
echo "Model: ${MODEL_PATH}"
echo "Nodes: ${NNODES}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "Total GPUs: $((NNODES * GPUS_PER_NODE))"
echo "Data home: ${RAY_DATA_HOME}"
echo "S3 Checkpoints: ${S3_CHECKPOINT_BASE}"
echo "Ray address: ${RAY_ADDRESS}"
echo "=================================="

# Submit Ray job
ray job submit --no-wait \
    --address "${RAY_ADDRESS}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=question \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.truncation=${truncation} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_prompt_micro_bsz_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optimizer_offload} \
    # MTC Configuration: S3 base path for tiered checkpoint storage
    # Checkpoints are first saved to CPU memory, then asynchronously synced to S3
    actor_rollout_ref.actor.checkpoint.s3_base_path=${S3_CHECKPOINT_BASE} \
    # MTC Configuration: Unique namespace for this training job
    # Used to organize checkpoints in S3 and enable recovery
    actor_rollout_ref.actor.checkpoint.ckpt_namespace=mtc-grpo-$(date +%s) \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_bsz_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_param_offload} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=1 \
    trainer.test_freq=2 \
    trainer.total_epochs=5 \
    # MTC Configuration: S3 base path for trainer-level checkpoints (e.g., critic)
    trainer.s3_base_path=${S3_CHECKPOINT_BASE}

echo ""
echo "Job submitted! Check status with: ray job status <job-id>"
echo "Or view logs with: ray job logs <job-id> --follow"
