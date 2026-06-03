#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -exuo pipefail

echo "=== NeMo RL GRPO Training via RayJob ==="
echo "Model: Qwen2.5-1.5B | GPUs: A10G | NVRx: enabled"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Hostname resolution: the rayjob.yaml uses `hostNetwork: true`, so the container
# shares the host's /etc/hosts (which already maps the node's hostname to its IP).
# We removed the legacy `echo >> /etc/hosts` line that ran here — it would have
# required either `privileged: true` or the `SYS_ADMIN` capability to succeed,
# and it is redundant on K8s with hostNetwork. If you adapt this entrypoint to
# run WITHOUT hostNetwork, pass `--node-ip-address=$MY_IP` to `ray start` or
# export `RAY_ADDRESS` instead of writing to /etc/hosts.
MY_IP=$(hostname -I | awk '{print $1}')
echo "Ray node IP: $MY_IP, hostname: $(hostname)"

# Apply NVRx runtime patches
if [ -f /shared/nvrx-demo/patches/patch_nvrx_features.py ]; then
    python3 /shared/nvrx-demo/patches/patch_nvrx_features.py
else
    echo "WARN: NVRx patches not found on FSx"
fi

# Optional full checkpoint wipe — set CLEAR_CHECKPOINTS=1 to start fresh every run
# Useful for demos where you want to show all training steps from step 1
CKPT_DIR="/shared/nvrx-demo/checkpoints"
if [ "${CLEAR_CHECKPOINTS:-0}" = "1" ]; then
    echo "INFO: CLEAR_CHECKPOINTS=1 — wiping $CKPT_DIR before training"
    rm -rf "$CKPT_DIR"/*
fi
mkdir -p "$CKPT_DIR"

# Cache HuggingFace models and datasets on FSx (download once, reuse across runs)
export HF_HOME="/shared/nvrx-demo/hf_cache"
export HF_DATASETS_CACHE="/shared/nvrx-demo/hf_datasets"
export TORCH_HOME="/shared/nvrx-demo/torch_cache"
export TRITON_CACHE_DIR="/shared/nvrx-demo/triton_cache"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR"
echo "Caches on FSx: HF_HOME=$HF_HOME (persistent across runs)"

cd /opt/nemo-rl

# Use NVRx wrapper if available, else standard run_grpo
SCRIPT="/shared/nvrx-demo/scripts/run_grpo_nvrx.py"
if [ ! -f "$SCRIPT" ]; then
    echo "WARN: NVRx wrapper not found, using examples/run_grpo.py"
    SCRIPT="examples/run_grpo.py"
fi
echo "Using: $SCRIPT"

# NOTE: ft_launcher is NOT used with RayJob on g5.
# ft_launcher's restart loop conflicts with KubeRay's cluster lifecycle
# and causes NCCL checkpoint deadlocks on A10G. Instead:
# - RayJob backoffLimit handles retries (fresh cluster per retry)
# - NVRx wrapper provides heartbeat monitoring + GPU health check
# - Checkpoints on FSx enable resume across RayJob retries
echo "=== Launching training (NVRx wrapper, no ft_launcher) ==="
echo "  Checkpointing: save_period=${CHECKPOINT_PERIOD:-10} to FSx"
echo "  NVRx: heartbeat + straggler detection via wrapper"

exec python3 "$SCRIPT" \
    cluster.num_nodes=2 \
    cluster.gpus_per_node=1 \
    grpo.max_num_steps="${GRPO_MAX_STEPS:-20}" \
    policy.model_name=Qwen/Qwen2.5-1.5B-Instruct \
    policy.tokenizer.name=Qwen/Qwen2.5-1.5B-Instruct \
    policy.megatron_cfg.enabled=false \
    policy.dtensor_cfg.enabled=true \
    policy.dtensor_cfg._v2=true \
    policy.dtensor_cfg.tensor_parallel_size=1 \
    policy.dtensor_cfg.activation_checkpointing=true \
    policy.dtensor_cfg.cpu_offload=true \
    policy.dtensor_cfg.lora_cfg.enabled=true \
    policy.dtensor_cfg.lora_cfg.dim=64 \
    policy.dtensor_cfg.lora_cfg.alpha=64 \
    policy.dtensor_cfg.lora_cfg.dropout=0.0 \
    policy.dtensor_cfg.lora_cfg.dropout_position=pre \
    policy.dtensor_cfg.lora_cfg.lora_A_init=xavier \
    'policy.dtensor_cfg.lora_cfg.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
    'policy.dtensor_cfg.lora_cfg.exclude_modules=[]' \
    policy.train_global_batch_size=32 \
    policy.train_micro_batch_size=2 \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=16 \
    policy.max_total_sequence_length=512 \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.4 \
    logger.wandb_enabled=false \
    logger.log_dir=/tmp/grpo-logs \
    logger.monitor_gpus=true \
    checkpointing.enabled=true \
    checkpointing.checkpoint_dir="$CKPT_DIR" \
    checkpointing.save_period="${CHECKPOINT_PERIOD:-10}" \
    checkpointing.keep_top_k="${CHECKPOINT_KEEP_TOP_K:-10}"
