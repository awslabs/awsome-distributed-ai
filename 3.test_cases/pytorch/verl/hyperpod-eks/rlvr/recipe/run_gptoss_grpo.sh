#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -xeuo pipefail

# ---------------------------------------------------------------------------
# veRL GRPO training for openai/gpt-oss-20b (Multilingual-Thinking)
#
# This script launches a GRPO training run via ``ray job submit`` against
# the existing Ray cluster.  It is designed for **g5.12xlarge** nodes
# (4 × A10G 24 GB, 1 × EFA) running the merged SFT bf16 checkpoint at
# /fsx/models/gpt-oss-20b-sft-merged/.
#
# ===== WHY THESE SETTINGS? (Lessons learned from 11 OOM iterations) =====
#
# 1. FSDP2 (not FSDP1):  FSDP1 explicitly disables CPUOffload for the
#    actor role (fsdp_workers.py: "causes incorrect results with grad
#    accumulation").  FSDP2 with offload_policy=True properly offloads
#    actor params to CPU during training, freeing GPU for vLLM rollout.
#
# 2. offload_policy=True:  This is the FSDP2-specific flag that enables
#    proper CPU offloading.  Without it, the ~20GB actor (bf16, 12 GPUs)
#    stays on GPU and leaves <4GB for vLLM — instant OOM on backward.
#
# 3. model_dtype=bf16:  veRL defaults the actor to fp32 in FSDP1 (but
#    keeps ref/critic at bf16).  Explicit bf16 halves memory.
#
# 4. gpu_memory_utilization=0.6:  This is the fraction of TOTAL GPU
#    memory for vLLM (model weights + KV cache), NOT just KV cache.
#    With a ~3.3GB model shard on each of 4 GPUs (TP=4), 0.6 × 23GB =
#    13.8GB allows ~10GB for KV cache.  Values below 0.5 leave less
#    than the model shard itself.
#
# 5. enforce_eager=True:  Disables CUDA graphs.  On A10G with tight
#    memory, CUDA graph capture allocates extra workspace that triggers
#    OOM.  The throughput cost is small since generation is not the
#    bottleneck.
#
# 6. use_kl_loss=False + use_kl_in_reward=False:  Disabling KL loss
#    removes the need for ref model log probs during the actor update,
#    reducing peak memory.  The ref model is still loaded for baseline
#    log prob computation during rollout.
#
# 7. max_response_length=256 (not 512):  Shorter completions reduce KV
#    cache pressure, response token memory, and log-prob computation.
#    For the language-compliance task, 256 tokens is sufficient.
#
# 8. nnodes=3 (not 4):  The Ray head pod does NOT have GPU resources in
#    K8s (no nvidia.com/gpu requested), so training uses only the 3
#    worker nodes.  Setting nnodes=4 would cause NCCL to hang waiting
#    for the head.
#
# 9. save_freq=20 + max_actor_ckpt_to_keep=3:  Each checkpoint is ~117GB
#    (full model state for 12 GPUs).  save_freq=1 filled a 1.2TB FSx
#    volume in 9 steps.  With save_freq=20 and max_keep=3, disk usage
#    stays under 351GB.
#
# 10. resume_mode=auto:  veRL reads latest_checkpointed_iteration.txt
#     and resumes from the last good checkpoint.  Critical for recovery
#     after crashes or preemptions.
#
# Key differences from the default run_grpo_configurable.sh:
#   • Model: openai/gpt-oss-20b merged SFT checkpoint (bf16, ~40 GB)
#   • Reward: custom language_reward.py  (language compliance, not math)
#   • Data:   HuggingFaceH4/Multilingual-Thinking parquet
#   • GPUs:   4 per node (g5.12xlarge), not 8 (p5en)
#   • Memory: FSDP2 + offload_policy for actor + ref offload
#   • vLLM:   TP=4 (one shard per GPU), gpu_memory_utilization=0.6
# ---------------------------------------------------------------------------

project_name='GRPO-GPT-OSS'
exp_name="GRPO-gpt-oss-20b-language"

# --- Algorithm ---------------------------------------------------------------
adv_estimator=grpo
use_kl_in_reward=False          # No KL penalty in reward (saves memory)
use_kl_loss=False               # No KL loss term (saves memory during actor update)
entropy_coeff=0

# --- Token lengths -----------------------------------------------------------
max_prompt_length=512
max_response_length=256         # 256 is enough for language-compliance task
filter_overlong_prompts=True
truncation=error

# --- Batch sizes -------------------------------------------------------------
# Total GPUs: 3 worker nodes × 4 GPUs = 12 GPUs
# Constraint: real_train_batch_size (train_batch_size × n) must be divisible
# by (n_gpus_per_node × nnodes).  12 × 2 = 24, 24 / 12 = 2 ✓
train_prompt_bsz=${TRAIN_BATCH_SIZE:-12}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-2}   # K=2 (conservative for memory)
train_prompt_mini_bsz=6                     # must be ≤ train_prompt_bsz
train_prompt_micro_bsz_per_gpu=1

# --- Ray & cluster -----------------------------------------------------------
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}

# NOTE: nnodes=3 because Ray head pod has no GPU resources in K8s.
# Only the 3 worker pods participate in FSDP/vLLM training.
NNODES=${NUM_NODES:-3}
GPUS_PER_NODE=${NUM_GPU_PER_NODE:-4}

# --- Model & data paths ------------------------------------------------------
MODEL_PATH=${MODEL_PATH:-"/fsx/models/gpt-oss-20b-sft-merged"}
RAY_DATA_HOME=${RAY_DATA_HOME:-"/fsx/verl"}
CKPTS_DIR="${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"

TRAIN_FILE="${RAY_DATA_HOME}/data/multilingual-thinking/train.parquet"
TEST_FILE="${RAY_DATA_HOME}/data/multilingual-thinking/test.parquet"

# --- Reward function ---------------------------------------------------------
# The reward file must be accessible on all Ray workers (e.g. on shared FSx).
REWARD_FN_PATH=${REWARD_FN_PATH:-"/fsx/verl/reward/language_reward.py"}
REWARD_FN_NAME=${REWARD_FN_NAME:-"compute_score"}

# --- Performance / memory ----------------------------------------------------
gen_tp=4                        # TP=4 — one model shard per GPU within a node
log_prob_micro_bsz_per_gpu=2    # Keep low to avoid OOM during log-prob pass
gpu_memory_utilization=0.6      # Fraction of TOTAL GPU memory for vLLM

# FSDP2 + full offloading — required for 20B MoE model on 24GB GPUs
actor_strategy=fsdp2            # FSDP2 supports offload_policy (FSDP1 does NOT)
model_dtype=bf16                # veRL defaults actor to fp32 — force bf16
param_offload=True              # Offload actor params to CPU
optimizer_offload=True          # Offload optimizer states to CPU
offload_policy=True             # FSDP2-specific: enables proper CPU offload
reshard_after_forward=True      # Free GPU memory after each forward pass
ref_param_offload=True          # Offload ref model params to CPU

# --- Checkpoint & resume ----------------------------------------------------
save_freq=${SAVE_FREQ:-20}              # Save every 20 steps (~1.1 hrs each)
test_freq=${TEST_FREQ:-20}              # Validate at same frequency as save
max_actor_ckpt_to_keep=3                # Keep max 3 checkpoints (~351GB)
total_epochs=${TOTAL_EPOCHS:-5}
resume_mode=${RESUME_MODE:-"auto"}      # Auto-resume from latest checkpoint

# --- Summary -----------------------------------------------------------------
echo "=== GRPO GPT-OSS-20B Configuration ==="
echo "Project       : ${project_name}"
echo "Experiment    : ${exp_name}"
echo "Model         : ${MODEL_PATH}"
echo "Nodes         : ${NNODES}"
echo "GPUs/node     : ${GPUS_PER_NODE}"
echo "Total GPUs    : $((NNODES * GPUS_PER_NODE))"
echo "Data          : ${TRAIN_FILE}"
echo "Reward        : ${REWARD_FN_PATH}::${REWARD_FN_NAME}"
echo "Checkpoints   : ${CKPTS_DIR}"
echo "Ray address   : ${RAY_ADDRESS}"
echo "Strategy      : ${actor_strategy}"
echo "Model dtype   : ${model_dtype}"
echo "TP            : ${gen_tp}"
echo "gpu_mem_util  : ${gpu_memory_utilization}"
echo "param_offload : ${param_offload}"
echo "optim_offload : ${optimizer_offload}"
echo "offload_policy: ${offload_policy}"
echo "ref_offload   : ${ref_param_offload}"
echo "save_freq     : ${save_freq}"
echo "max_ckpt_keep : ${max_actor_ckpt_to_keep}"
echo "resume_mode   : ${resume_mode}"
echo "======================================="

# --- Submit ------------------------------------------------------------------
ray job submit --no-wait \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.truncation=${truncation} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=${actor_strategy} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_prompt_micro_bsz_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optimizer_offload} \
    actor_rollout_ref.actor.fsdp_config.offload_policy=${offload_policy} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=${model_dtype} \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=${reshard_after_forward} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.load_format=dummy_dtensor \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_bsz_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_param_offload} \
    actor_rollout_ref.ref.fsdp_config.model_dtype=${model_dtype} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name="${REWARD_FN_NAME}" \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.save_freq=${save_freq} \
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.resume_mode=${resume_mode}
