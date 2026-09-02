#!/usr/bin/env bash
set -xeuo pipefail

# Project configuration
project_name='GRPO'
exp_name="GRPO-${MODEL_NAME}"

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
train_prompt_bsz=${TRAIN_BATCH_SIZE:-32}  # Reduced from 256 for faster testing
gen_prompt_bsz=${GEN_BATCH_SIZE:-$train_prompt_bsz}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-2}  # Reduced from 5 for faster testing
train_prompt_mini_bsz=16  # Must be <= train_prompt_bsz
train_prompt_micro_bsz_per_gpu=1

# Ray configuration from env_vars
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}

# Cluster configuration from env_vars
NNODES=${NUM_NODES:-4}
GPUS_PER_NODE=${NUM_GPU_PER_NODE:-8}

# Model and data paths from env_vars
MODEL_NAME=${MODEL_NAME:-"Qwen3-8B"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
RAY_DATA_HOME=${RAY_DATA_HOME:-"/fsx/verl"}
CKPTS_DIR="${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"

# Data files - using GSM8K dataset
TRAIN_FILE="${RAY_DATA_HOME}/data/gsm8k/train.parquet"
TEST_FILE="${RAY_DATA_HOME}/data/gsm8k/test.parquet"

# Performance parameters
gen_tp=2
log_prob_micro_bsz_per_gpu=32
gpu_memory_utilization=0.6

# Memory optimization
param_offload=False
optimizer_offload=False
ref_param_offload=True

# Print configuration for verification
echo "=== GRPO Training Configuration ==="
echo "Project: ${project_name}"
echo "Experiment: ${exp_name}"
echo "Model: ${MODEL_NAME} (${MODEL_PATH})"
echo "Nodes: ${NNODES}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "Total GPUs: $((NNODES * GPUS_PER_NODE))"
echo "Data home: ${RAY_DATA_HOME}"
echo "Checkpoints: ${CKPTS_DIR}"
echo "Ray address: ${RAY_ADDRESS}"
echo "=================================="

# Submit Ray job
ray job submit --no-wait \
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
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.save_freq=1 \
    trainer.test_freq=2 \
    trainer.total_epochs=2 
