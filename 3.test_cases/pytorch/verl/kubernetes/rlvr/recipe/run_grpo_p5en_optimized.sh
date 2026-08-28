#!/usr/bin/env bash
set -xeuo pipefail

# =============================================================================
# Optimized GRPO recipe for p5en.48xlarge (8x H200 per node)
#
# Key optimizations over run_grpo_configurable.sh:
#   1. Dynamic batching (use_dynamic_bsz=True) — eliminates padding waste in
#      actor updates. Typical GSM8K sequences average ~317 tokens vs 1024 max,
#      so fixed-size batching wastes ~69% of compute on padding.
#   2. FSDP2 (strategy=fsdp2) — PyTorch's next-gen fully sharded data
#      parallelism with per-parameter sharding. ~7% lower memory, ~1.5%
#      throughput gain over FSDP1.
#   3. Forward prefetch (forward_prefetch=True) — overlaps FSDP all-gather
#      with computation for pipelined communication/compute.
#   4. Higher vLLM KV cache (gpu_memory_utilization=0.7) — more cache for
#      faster generation batching.
#
# IMPORTANT: PYTORCH_CUDA_ALLOC_CONF must be empty in the runtime env to
# avoid conflicts with vLLM v1's CuMemAllocator. verl internally toggles
# expandable_segments on/off at the training/inference boundary.
#
# Tested with: verl v0.6.1, 4x p5en.48xlarge (32 H200 GPUs), Qwen3-8B
# =============================================================================

# Project configuration
project_name='GRPO'
exp_name="GRPO-${MODEL_NAME}-optimized"

# GRPO Algorithm parameters
adv_estimator=grpo
use_kl_in_reward=False
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl
entropy_coeff=0

# Token length configuration
max_prompt_length=512
max_response_length=512
filter_overlong_prompts=True
truncation='error'

# Training configuration
train_prompt_bsz=${TRAIN_BATCH_SIZE:-64}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
train_prompt_mini_bsz=16  # Must be <= train_prompt_bsz

# Dynamic batching: pack sequences by total token count instead of fixed
# micro-batch size. ppo_max_token_len_per_gpu should be >= 2x the max total
# sequence length (2 * 1024 = 2048). Setting 4096 provides headroom.
# When use_dynamic_bsz=True, ppo_micro_batch_size_per_gpu is ignored.
use_dynamic_bsz=True
ppo_max_token_len_per_gpu=4096

# FSDP2 + forward prefetch
strategy=fsdp2
forward_prefetch=True

# Ray configuration from env_vars
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}

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
gpu_memory_utilization=0.7  # Higher than default 0.6 for more KV cache

# Memory optimization
param_offload=False
optimizer_offload=False
ref_param_offload=True

# Checkpoint configuration
save_freq=20
test_freq=5

# Print configuration for verification
echo "=== GRPO Optimized Training Configuration ==="
echo "Project: ${project_name}"
echo "Experiment: ${exp_name}"
echo "Model: ${MODEL_NAME} (${MODEL_PATH})"
echo "Nodes: ${NNODES}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "Total GPUs: $((NNODES * GPUS_PER_NODE))"
echo "Data home: ${RAY_DATA_HOME}"
echo "Checkpoints: ${CKPTS_DIR}"
echo "Ray address: ${RAY_ADDRESS}"
echo "--- Optimizations ---"
echo "Dynamic batching: ${use_dynamic_bsz} (max_token_len=${ppo_max_token_len_per_gpu})"
echo "Strategy: ${strategy}"
echo "Forward prefetch: ${forward_prefetch}"
echo "GPU memory utilization: ${gpu_memory_utilization}"
echo "================================================"

# Submit Ray job
# NOTE: PYTORCH_CUDA_ALLOC_CONF="" overrides the pod-level env var to prevent
# vLLM v1 crash (AssertionError: Expandable segments not compatible with memory pool).
# verl manages expandable_segments internally at the training/inference boundary.
ray job submit --no-wait \
    --address "${RAY_ADDRESS}" \
    --runtime-env-json '{"env_vars": {"NCCL_DEBUG": "INFO", "TOKENIZERS_PARALLELISM": "false", "HYDRA_FULL_ERROR": "1", "PYTORCH_CUDA_ALLOC_CONF": ""}}' \
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
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optimizer_offload} \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=${forward_prefetch} \
    actor_rollout_ref.actor.strategy=${strategy} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_bsz_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_param_offload} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    trainer.critic_warmup=0 \
    'trainer.logger=["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=2
