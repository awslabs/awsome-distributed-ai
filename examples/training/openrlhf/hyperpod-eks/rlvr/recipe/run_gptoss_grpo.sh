#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -xeuo pipefail

# ---------------------------------------------------------------------------
# OpenRLHF GRPO training for openai/gpt-oss-20b (Multilingual-Thinking)
#
# This script launches a GRPO training run via ``ray job submit`` against
# the existing Ray cluster.  It is designed for **g5.12xlarge** nodes
# (4 x A10G 24GB, 1 x EFA) running the merged SFT bf16 checkpoint at
# /fsx/models/gpt-oss-20b-sft-merged/.
#
# ===== ARCHITECTURE =====
#
# OpenRLHF uses a fundamentally different architecture from veRL:
#   - DeepSpeed ZeRO-3 for training (vs veRL's FSDP2)
#   - vLLM for inference on dedicated GPUs (same engine, different placement)
#   - Non-Hybrid Engine: vLLM and training on SEPARATE GPU nodes
#   - Ray orchestrates everything (same as veRL)
#
# ===== KEY MEMORY OPTIMIZATIONS =====
#
# 1. Non-Hybrid Engine (SEPARATE vLLM + training nodes):
#    vLLM inference runs on dedicated GPUs (1 node, 4 GPUs, TP=4).
#    DeepSpeed training runs on separate GPUs (3 nodes, 12 GPUs).
#    This avoids the sleep/wake CPU backup overhead that causes OOM
#    when vLLM + DeepSpeed colocate on the same GPUs.
#
#    We tried Hybrid Engine (--colocate_all_models) but on 24GB A10Gs:
#    - With adam_offload: CPU OOM (optimizer ~107GB + vLLM sleep ~40GB > 160Gi)
#    - Without adam_offload: GPU OOM during backward pass (17.4GB used of 22GB)
#
# 2. DeepSpeed ZeRO-3 + adam_offload:
#    Model parameters, gradients, and optimizer states are sharded across
#    16 training GPUs.  Optimizer states offloaded to CPU.
#    Per-node CPU: 320GB/16GPUs * 4GPUs/node = ~80GB + ~10GB overhead = ~90GB.
#    Fits comfortably in 160Gi pod limit with ~70GB headroom for checkpoints.
#
# 3. No critic model:
#    GRPO is a critic-free algorithm.  Advantage is estimated as
#    (reward - mean) / std within each prompt group.  No critic = no extra
#    GPU memory for a second model.
#
# 4. enforce_eager=True:
#    Disables CUDA graphs in vLLM.  On A10G with tight memory, CUDA graph
#    capture allocates extra workspace that triggers OOM.
#
# 5. init_kl_coef=0:
#    Disabling KL penalty means no ref model forward pass during training,
#    reducing peak memory.  (We can add KL back if training is unstable.)
#
# 6. vllm_gpu_memory_utilization=0.8:
#    vLLM has dedicated GPUs (no training competing), so we can allocate
#    more GPU memory for KV cache, improving generation throughput.
#
# ===== DIFFERENCES FROM veRL =====
#
# | Aspect              | veRL                    | OpenRLHF              |
# |---------------------|------------------------|-----------------------|
# | Training framework  | FSDP2                  | DeepSpeed ZeRO-3     |
# | Memory offload      | offload_policy=True    | --adam_offload        |
# | Inference           | vLLM (inline)          | vLLM (dedicated node)|
# | GPU layout          | All GPUs shared        | Separate inf/train   |
# | Checkpoint format   | FSDP shards → merge    | HF format directly   |
# | Reward function API | compute_score()        | reward_func()        |
# | KL handling         | use_kl_loss=False      | --init_kl_coef 0     |
# | MoE balancing       | aux_loss_coef (hydra)  | --aux_loss_coef      |
# ---------------------------------------------------------------------------

# --- Environment defaults ----------------------------------------------------
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}

# NOTE: nnodes=4 for TRAINING nodes.  With the 6-node setup:
#   - Ray head: 8Gi CPU, num-gpus=0 (co-located with 1 worker on same node)
#   - 5 workers: 160Gi CPU, 4 GPUs, 1 EFA each
#   - 1 worker: vLLM inference (TP=4, dedicated)
#   - 4 workers: DeepSpeed training with adam_offload
# adam_offload: 320GB / 16 GPUs × 4 GPUs/node = 80GB/node → plenty of headroom
NNODES=${NUM_NODES:-4}
GPUS_PER_NODE=${NUM_GPU_PER_NODE:-4}
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))

# --- Model & data paths ------------------------------------------------------
MODEL_PATH=${MODEL_PATH:-"/fsx/models/gpt-oss-20b-sft-merged"}
RAY_DATA_HOME=${RAY_DATA_HOME:-"/fsx/openrlhf"}
SAVE_PATH="${RAY_DATA_HOME}/checkpoints/grpo-gpt-oss-20b"
CKPT_PATH="${SAVE_PATH}/ckpt"

TRAIN_DATA="${RAY_DATA_HOME}/data/multilingual-thinking/train.jsonl"

# --- Reward function ---------------------------------------------------------
REWARD_FN_PATH=${REWARD_FN_PATH:-"/fsx/openrlhf/reward/language_reward.py"}

# --- Algorithm ---------------------------------------------------------------
ADVANTAGE_ESTIMATOR="group_norm"        # GRPO: (reward - mean) / std
INIT_KL_COEF=0                          # No KL penalty (saves memory)
# If training is unstable, try:
#   INIT_KL_COEF=1e-3  with  --use_kl_loss --kl_estimator k3

# --- Batch sizes -------------------------------------------------------------
# Total GPUs: 4 nodes x 4 GPUs = 16 GPUs
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}        # rollout_batch * n_samples
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2} # K=2 for GRPO
MICRO_TRAIN_BATCH_SIZE=${MICRO_TRAIN_BATCH_SIZE:-1}
MICRO_ROLLOUT_BATCH_SIZE=${MICRO_ROLLOUT_BATCH_SIZE:-1}

# --- Token lengths -----------------------------------------------------------
PROMPT_MAX_LEN=512
GENERATE_MAX_LEN=256                    # 256 is enough for language-compliance task

# --- vLLM (Dedicated Inference Node) -----------------------------------------
# Non-Hybrid Engine: vLLM gets its own dedicated worker (4 GPUs, TP=4).
# Training (DeepSpeed ZeRO-3) runs on the other 4 workers (16 GPUs).
#
# Resource allocation (20 total GPUs across 5 worker nodes):
#   vLLM:  1 engine × TP=4 = 4 GPUs  (1 worker node, dedicated inference)
#   Actor: 4 nodes × 4 GPUs/node = 16 GPUs (workers with 160Gi memory)
#   Head:  8Gi CPU, num-gpus=0 (orchestration only, co-located with a worker)
#
# adam_offload per training node: 320GB / 16 GPUs × 4 GPUs = 80GB
# With 160Gi pod limit, that leaves ~80GB headroom for checkpointing.
ACTOR_NUM_NODES=${NNODES}
ACTOR_NUM_GPUS_PER_NODE=${GPUS_PER_NODE}
# ref_num not used when init_kl_coef=0 (no ref model)
REF_NUM_NODES=${NNODES}
REF_NUM_GPUS_PER_NODE=${GPUS_PER_NODE}
VLLM_NUM_ENGINES=1                      # One vLLM engine on dedicated node
VLLM_TP=4                              # TP=4 within that node (one shard per GPU)
VLLM_GPU_MEM_UTIL=0.8                  # Higher util since no training competes

# --- Checkpoint & resume ----------------------------------------------------
SAVE_STEPS=${SAVE_STEPS:-20}
MAX_CKPT_NUM=${MAX_CKPT_NUM:-3}
NUM_EPISODES=${NUM_EPISODES:-1}
MAX_SAMPLES=${MAX_SAMPLES:-1280}        # 80 steps * 16 rollout_batch_size
MAX_EPOCHS=1

# --- MoE balancing (gpt-oss-20b is a MoE model) ----------------------------
AUX_LOSS_COEF=0.01                      # MoE load balancing loss

# --- Summary -----------------------------------------------------------------
echo "=== OpenRLHF GRPO GPT-OSS-20B Configuration ==="
echo "Model           : ${MODEL_PATH}"
echo "Nodes           : ${NNODES}"
echo "GPUs/node       : ${GPUS_PER_NODE}"
echo "Total GPUs      : ${TOTAL_GPUS}"
echo "Data            : ${TRAIN_DATA}"
echo "Reward function : ${REWARD_FN_PATH}"
echo "Save path       : ${SAVE_PATH}"
echo "Ray address     : ${RAY_ADDRESS}"
echo "Advantage       : ${ADVANTAGE_ESTIMATOR}"
echo "KL coef         : ${INIT_KL_COEF}"
echo "Rollout batch   : ${ROLLOUT_BATCH_SIZE}"
echo "Train batch     : ${TRAIN_BATCH_SIZE}"
echo "N samples/prompt: ${N_SAMPLES_PER_PROMPT}"
echo "vLLM engines    : ${VLLM_NUM_ENGINES} x TP=${VLLM_TP}"
echo "vLLM GPU mem    : ${VLLM_GPU_MEM_UTIL}"
echo "MoE aux_loss    : ${AUX_LOSS_COEF}"
echo "Save steps      : ${SAVE_STEPS}"
echo "Max checkpoints : ${MAX_CKPT_NUM}"
echo "================================================"

# --- Submit ------------------------------------------------------------------
ray job submit \
    --address "${RAY_ADDRESS}" \
    --no-wait \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --pretrain "${MODEL_PATH}" \
    --save_path "${SAVE_PATH}" \
    --ckpt_path "${CKPT_PATH}" \
    --save_hf_ckpt \
    --load_checkpoint \
    --max_ckpt_num ${MAX_CKPT_NUM} \
    --save_steps ${SAVE_STEPS} \
    \
    --remote_rm_url "${REWARD_FN_PATH}" \
    --prompt_data "${TRAIN_DATA}" \
    --input_key prompt \
    --label_key label \
    --apply_chat_template \
    --prompt_max_len ${PROMPT_MAX_LEN} \
    --generate_max_len ${GENERATE_MAX_LEN} \
    \
    --advantage_estimator ${ADVANTAGE_ESTIMATOR} \
    --init_kl_coef ${INIT_KL_COEF} \
    --aux_loss_coef ${AUX_LOSS_COEF} \
    --actor_learning_rate 1e-6 \
    --lr_warmup_ratio 0.03 \
    --max_norm 1.0 \
    --temperature 0.7 \
    --top_p 0.95 \
    --entropy_loss_coef 0.0 \
    \
    --rollout_batch_size ${ROLLOUT_BATCH_SIZE} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --n_samples_per_prompt ${N_SAMPLES_PER_PROMPT} \
    --micro_train_batch_size ${MICRO_TRAIN_BATCH_SIZE} \
    --micro_rollout_batch_size ${MICRO_ROLLOUT_BATCH_SIZE} \
    --num_episodes ${NUM_EPISODES} \
    --max_epochs ${MAX_EPOCHS} \
    --max_samples ${MAX_SAMPLES} \
    \
    --actor_num_nodes ${ACTOR_NUM_NODES} \
    --actor_num_gpus_per_node ${ACTOR_NUM_GPUS_PER_NODE} \
    --ref_num_nodes ${REF_NUM_NODES} \
    --ref_num_gpus_per_node ${REF_NUM_GPUS_PER_NODE} \
    --vllm_num_engines ${VLLM_NUM_ENGINES} \
    --vllm_tensor_parallel_size ${VLLM_TP} \
    --vllm_gpu_memory_utilization ${VLLM_GPU_MEM_UTIL} \
    --vllm_sync_backend nccl \
    --enforce_eager \
    \
    --zero_stage 3 \
    --adam_offload \
    --gradient_checkpointing \
    --bf16 \
    --packing_samples \
    \
    --logging_steps 1 \
    --seed 42
