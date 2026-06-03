#!/usr/bin/env bash
set -xeuo pipefail

# Source environment variables if not already loaded
if [ -z "${MODEL_NAME:-}" ]; then
    echo "Loading environment variables from setup/env_vars..."
    source ../../setup/env_vars
fi

# Project configuration
project_name='DAPO'
exp_name="DAPO-${MODEL_NAME}"

# DAPO Algorithm parameters
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

# Token length configuration
# max_prompt_length=$((1024 * 2))
max_prompt_length=512
# max_response_length=$((1024 * 20))
max_response_length=1024
enable_overlong_buffer=True
overlong_buffer_len=512  # Must be less than max_response_length
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

# Training configuration
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=0  # 0 = unlimited retries, or set to higher value like 10
train_prompt_bsz=${TRAIN_BATCH_SIZE:-512}
gen_prompt_bsz=${GEN_BATCH_SIZE:-$((train_prompt_bsz * 3))}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-16}
train_prompt_mini_bsz=32

# Ray configuration from env_vars
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
RUNTIME_ENV="${WORKING_DIR}/recipe/dapo/runtime_env.yaml"


# Cluster configuration from env_vars
NNODES=${NUM_NODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Model and data paths from env_vars
MODEL_NAME=${MODEL_NAME:-"Qwen3-8B"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
RAY_DATA_HOME=${RAY_DATA_HOME:-"/fsx/verl"}
CKPTS_DIR="${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"
TRAIN_FILE="${RAY_DATA_HOME}/data/dapo-math-17k.parquet"
TEST_FILE="${RAY_DATA_HOME}/data/aime-2024.parquet"

# Algorithm parameters
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance parameters (adjust based on your cluster size)
sp_size=8
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=4

# Print configuration for verification
echo "=== DAPO Training Configuration ==="
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
ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto