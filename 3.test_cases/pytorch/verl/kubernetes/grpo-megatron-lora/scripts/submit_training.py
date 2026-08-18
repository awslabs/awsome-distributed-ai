#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
GRPO Training Job Submitter — Hydra-powered configuration.

Replaces env_vars + shell scripts with a single composable config system.
Loads Hydra config groups, resolves all interpolations, and submits a Ray job
with the proper verl CLI overrides for either the FSDP or Megatron backend.

Usage:
    # Default: Qwen3-8B, FSDP backend, 6-node, LoRA enabled, MLflow tracking
    python3 scripts/submit_training.py

    # Megatron backend with large model
    python3 scripts/submit_training.py backend=megatron model=qwen25-72b

    # Quick validation on 1 node, no MLflow
    python3 scripts/submit_training.py compute.num_nodes=1 compute.agent_num_workers=4 tracking=console

    # Switch to target model
    python3 scripts/submit_training.py model=qwen3-coder-next

    # Tune hyperparameters
    python3 scripts/submit_training.py training.learning_rate=1e-4 training.train_batch_size=128

    # Full fine-tuning (no LoRA)
    python3 scripts/submit_training.py lora=disabled

    # Dry run (print config and command, don't submit)
    python3 scripts/submit_training.py --cfg job
"""

import json
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Compile-cache namespace for the current Docker/runtime stack.  Bump when the
# base image's torch/vLLM/CUDA changes (see cache dir comment in
# build_verl_overrides).  "verl-v0.8.0" = torch 2.11 / vLLM 0.20.2 / CUDA 13.
CACHE_STACK_VERSION = "verl-v0.8.0"


# =============================================================================
# Shared overrides — common to both FSDP and Megatron backends
# =============================================================================
def _build_shared_overrides(cfg: DictConfig) -> list[str]:
    """Build verl overrides that are identical across backends."""
    t = cfg.training
    max_token_len = t.max_prompt_length + t.max_response_length

    overrides = [
        # Algorithm
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        # Data
        f"data.train_files={cfg.data_paths.train_file}",
        f"data.val_files={cfg.data_paths.val_file}",
        f"data.prompt_key={cfg.data_paths.prompt_key}",
        f"data.train_batch_size={t.train_batch_size}",
        f"data.max_prompt_length={t.max_prompt_length}",
        f"data.max_response_length={t.max_response_length}",
        "data.filter_overlong_prompts=True",
        f"data.truncation={t.data_truncation}",
        # Model
        f"actor_rollout_ref.model.path={cfg.model.fsx_path}",
        f"actor_rollout_ref.model.use_shm={cfg.model.use_shm}",
        # Actor optimizer
        f"actor_rollout_ref.actor.optim.lr={t.learning_rate}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps={t.lr_warmup_steps}",
        f"actor_rollout_ref.actor.optim.weight_decay={t.weight_decay}",
        # Actor PPO
        # Per-model ppo_mini_batch_size override — MoE models with non-standard
        # DP sizes need a value that satisfies the Megatron normalization:
        #   effective = config_value * rollout_n // dp_world_size
        #   per_dp_batch % effective == 0
        # Falls back to the training config default when model doesn't override.
        f"actor_rollout_ref.actor.ppo_mini_batch_size="
        f"{OmegaConf.select(cfg.model, 'ppo_mini_batch_size', default=t.ppo_mini_batch_size)}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={t.ppo_micro_batch_size_per_gpu}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={t.kl_loss_coef}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        # Clipping (DAPO-style)
        "actor_rollout_ref.actor.clip_ratio_low=0.2",
        "actor_rollout_ref.actor.clip_ratio_high=0.28",
        "actor_rollout_ref.actor.clip_ratio_c=10.0",
        # Dynamic batching
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={max_token_len}",
        # Rollout (vLLM)
        # Use model-specific rollout TP if set (e.g. MoE models use full-node TP=8
        # for vLLM inference while actor TP=2), otherwise fall back to actor TP.
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size="
        f"{OmegaConf.select(cfg.model, 'rollout_tensor_parallel_size', default=cfg.model.tensor_parallel_size)}",
        # Per-model gpu_memory_utilization override (MoE models need lower values
        # for expert routing headroom), falls back to the compute default.
        f"actor_rollout_ref.rollout.gpu_memory_utilization="
        f"{OmegaConf.select(cfg.model, 'gpu_memory_utilization', default=cfg.compute.gpu_memory_utilization)}",
        f"actor_rollout_ref.rollout.n={t.n_responses_per_prompt}",
        f"actor_rollout_ref.rollout.max_num_seqs={t.max_num_seqs}",
        f"actor_rollout_ref.rollout.max_model_len={max_token_len}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={max_token_len}",
        f"actor_rollout_ref.rollout.enforce_eager="
        f"{OmegaConf.select(cfg.model, 'enforce_eager', default=True)}",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.top_k=-1",
        # Rollout validation kwargs
        "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
        "actor_rollout_ref.rollout.val_kwargs.top_p=0.7",
        "actor_rollout_ref.rollout.val_kwargs.top_k=-1",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
        # Reference model
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        # NCCL timeout
        "actor_rollout_ref.nccl_timeout=1200",
        # Trainer
        "trainer.critic_warmup=0",
        f"trainer.n_gpus_per_node={cfg.compute.gpus_per_node}",
        f"trainer.nnodes={cfg.compute.num_nodes}",
        f"trainer.default_local_dir={cfg.compute.fsx_home}/ckpts/{cfg.data_name}/{cfg.backend.name}/{cfg.model.name}",
        f"trainer.save_freq={t.save_freq}",
        f"trainer.test_freq={t.test_freq}",
        f"trainer.total_epochs={t.total_epochs}",
        # Run in-training validation ONCE before the first optimizer step.
        # Default False (it costs a full val pass), but essential for measuring a
        # BASE / step-0 held-out number on exactly the same instrument as the
        # step-N validations. With LoRA the B matrices are zero-initialised, so
        # the step-0 model is bit-identical to the base model — which makes this
        # the only way to get a base val-core number that is directly comparable
        # to the step-50/100/150 numbers. Without it, every base baseline has to
        # come from a separate harness and is not comparable.
        f"trainer.val_before_train={bool(OmegaConf.select(t, 'val_before_train', default=False))}",
    ]

    return overrides


# =============================================================================
# FSDP-specific overrides
# =============================================================================
def _build_fsdp_overrides(cfg: DictConfig) -> list[str]:
    """Build verl overrides specific to the FSDP backend."""
    overrides = [
        # FSDP model flags
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        # FSDP normalizes ppo_mini_batch_size per-GPU as:
        #   (mini_batch * rollout.n) // world_size
        # With 48 GPUs and n=4, the shared default (16) yields 64//48=1,
        # which isn't divisible by micro_batch=2.  Use 24 → 96//48=2.
        "actor_rollout_ref.actor.ppo_mini_batch_size=24",
        # FSDP sharding config
        f"actor_rollout_ref.actor.fsdp_config.param_offload={cfg.compute.param_offload}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={cfg.compute.optimizer_offload}",
        f"actor_rollout_ref.actor.fsdp_config.fsdp_size={cfg.compute.gpus_per_node}",
        # Agent loop workers
        f"actor_rollout_ref.rollout.agent.num_workers={cfg.compute.agent_num_workers}",
        # Reference model (FSDP offloads ref params to CPU)
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        # Disable torch.compile for FSDP — verl 0.8.0.dev defaults to True,
        # which causes NCCL _ALLGATHER_BASE timeouts at step 3 due to
        # recompilation hangs with dynamic sequence lengths.  The v0.7.0
        # default was False and previous FSDP runs succeeded without it.
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False",
        "actor_rollout_ref.ref.fsdp_config.use_torch_compile=False",
        # Trainer naming
        "trainer.project_name=GRPO-FSDP",
        f"trainer.experiment_name=GRPO-FSDP-{cfg.model.name}",
    ]

    # LoRA (FSDP uses HuggingFace PEFT)
    if cfg.lora.enabled:
        overrides.extend(
            [
                f"actor_rollout_ref.model.lora_rank={cfg.lora.rank}",
                f"actor_rollout_ref.model.lora_alpha={cfg.lora.alpha}",
                "actor_rollout_ref.rollout.load_format=safetensors",
                f"actor_rollout_ref.rollout.layered_summon={cfg.lora.layered_summon}",
            ]
        )
        # target_modules: "all-linear" is a PEFT magic string (quoted scalar),
        # comma-separated names are a Hydra list
        tm = cfg.lora.target_modules
        if "," in str(tm):
            overrides.append(f"actor_rollout_ref.model.target_modules=[{tm}]")
        else:
            overrides.append(f"actor_rollout_ref.model.target_modules='{tm}'")

    return overrides


# =============================================================================
# Megatron-specific overrides
# =============================================================================
def _build_megatron_overrides(cfg: DictConfig) -> list[str]:
    """Build verl overrides specific to the Megatron backend."""
    max_token_len = cfg.training.max_prompt_length + cfg.training.max_response_length

    overrides = [
        # Model-level fused kernels flag — upstream 235B reference sets True.
        # Megatron also applies granular fusions via override_transformer_config below.
        f"actor_rollout_ref.model.use_fused_kernels={cfg.backend.use_fused_kernels}",
        # Megatron actor parallelism
        f"actor_rollout_ref.actor.megatron.tensor_model_parallel_size={cfg.model.tensor_parallel_size}",
        f"actor_rollout_ref.actor.megatron.pipeline_model_parallel_size={cfg.model.pipeline_parallel_size}",
        f"actor_rollout_ref.actor.megatron.expert_model_parallel_size={cfg.model.expert_parallel_size}",
        f"actor_rollout_ref.actor.megatron.expert_tensor_parallel_size={cfg.backend.expert_tensor_parallel_size}",
        f"actor_rollout_ref.actor.megatron.context_parallel_size={cfg.model.context_parallel_size}",
        # Megatron offloading — per-model overrides allow 235B to enable offloading
        # even though the compute defaults are False (sufficient for smaller models).
        f"actor_rollout_ref.actor.megatron.param_offload="
        f"{OmegaConf.select(cfg.model, 'param_offload', default=cfg.compute.param_offload)}",
        f"actor_rollout_ref.actor.megatron.optimizer_offload="
        f"{OmegaConf.select(cfg.model, 'optimizer_offload', default=cfg.compute.optimizer_offload)}",
        f"actor_rollout_ref.actor.megatron.grad_offload="
        f"{OmegaConf.select(cfg.model, 'grad_offload', default=cfg.backend.grad_offload)}",
        # Megatron-specific actor flags
        f"actor_rollout_ref.actor.optim.clip_grad={cfg.backend.clip_grad}",
        # Restore the LR/optimizer param scheduler from the checkpoint on resume.
        # verl defaults this to False, which means every resume RE-RUNS LR warmup
        # from 0 instead of continuing the schedule — a resumed run silently spends
        # its first `lr_warmup_steps` at a fraction of the configured LR.
        f"actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler="
        f"{cfg.backend.use_checkpoint_opt_param_scheduler}",
        f"actor_rollout_ref.actor.loss_agg_mode={cfg.backend.loss_agg_mode}",
        # Megatron-specific rollout flags
        "actor_rollout_ref.rollout.load_format=dummy",
        f"actor_rollout_ref.rollout.free_cache_engine={cfg.backend.free_cache_engine}",
        f"actor_rollout_ref.rollout.enable_chunked_prefill={cfg.backend.enable_chunked_prefill}",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={max_token_len}",
        # Reference model — mirror actor parallelism
        f"actor_rollout_ref.ref.megatron.tensor_model_parallel_size={cfg.model.tensor_parallel_size}",
        f"actor_rollout_ref.ref.megatron.pipeline_model_parallel_size={cfg.model.pipeline_parallel_size}",
        f"actor_rollout_ref.ref.megatron.expert_model_parallel_size={cfg.model.expert_parallel_size}",
        f"actor_rollout_ref.ref.megatron.expert_tensor_parallel_size={cfg.backend.expert_tensor_parallel_size}",
        f"actor_rollout_ref.ref.megatron.context_parallel_size={cfg.model.context_parallel_size}",
        # Reference model — always offload ref params to CPU.
        # Uses a SEPARATE key (ref_param_offload) from the actor's param_offload
        # to avoid the actor optimization (param_offload=False) accidentally
        # disabling ref offloading.  The ref is a frozen copy used only for KL
        # divergence — keeping it on GPU wastes ~41 GB and can cause OOM/NCCL
        # errors during DDP broadcast_params().  Mirrors FSDP behavior (line 153)
        # which hardcodes ref param_offload=True.
        f"actor_rollout_ref.ref.megatron.param_offload="
        f"{OmegaConf.select(cfg.model, 'ref_param_offload', default=True)}",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={max_token_len}",
        # Trainer naming and resume
        "trainer.project_name=GRPO-Megatron",
        f"trainer.experiment_name=GRPO-Megatron-{cfg.model.name}",
        f"trainer.resume_mode={cfg.backend.resume_mode}",
    ]

    # Explicit resume path (only meaningful with resume_mode=resume_path).
    # Lets us roll back to a known-good checkpoint instead of the latest one.
    resume_from_path = OmegaConf.select(cfg.backend, "resume_from_path", default=None)
    if resume_from_path:
        overrides.append(f"trainer.resume_from_path={resume_from_path}")

    # Megatron-Bridge LoRA (native Megatron LoRA, not PEFT)
    if cfg.lora.enabled:
        lora_mg = cfg.lora.megatron
        # Per-model LoRA alpha override (e.g. Qwen3-30B-A3B uses alpha=64
        # vs the global default of 32). Falls back to lora config default.
        lora_alpha = OmegaConf.select(cfg.model, "lora_alpha", default=cfg.lora.alpha)
        # Per-model LoRA target_modules override — MoE models must exclude
        # expert layers (linear_fc1, linear_fc2) due to a vLLM LoRA naming
        # mismatch (w13.weight → w13_base_layer.weight). See verl#5479.
        lora_targets = OmegaConf.select(
            cfg.model, "lora_target_modules", default=lora_mg.target_modules
        )
        # Per-model LoRA merge override — MoE models need merge=True so that
        # vLLMHttpServer disables --enable_lora (preventing vLLM from wrapping
        # all supported modules with LoRA adapters).  Megatron-Bridge merges
        # LoRA into base weights before export_weights().
        lora_merge = OmegaConf.select(cfg.model, "lora_merge", default=lora_mg.merge)
        overrides.extend(
            [
                f"actor_rollout_ref.model.lora.type={lora_mg.type}",
                f"actor_rollout_ref.model.lora.rank={cfg.lora.rank}",
                f"actor_rollout_ref.model.lora.alpha={lora_alpha}",
                f"actor_rollout_ref.model.lora.dropout={lora_mg.dropout}",
                f"actor_rollout_ref.model.lora.merge={lora_merge}",
                f"actor_rollout_ref.model.lora.target_modules=[{lora_targets}]",
                f"actor_rollout_ref.actor.megatron.use_mbridge={cfg.backend.use_mbridge}",
                f"actor_rollout_ref.actor.megatron.vanilla_mbridge={cfg.backend.vanilla_mbridge}",
                # Native HF PEFT adapter export at every checkpoint save.
                # With 'hf_model' in save_contents and peft_cls set, verl v0.8.0's
                # MegatronCheckpointManager calls bridge.save_hf_adapter() to write a
                # standard PEFT adapter to <ckpt>/actor/huggingface/adapter/. This is
                # cheap for LoRA (adapter tensors only) and makes checkpoints directly
                # mergeable via peft merge_and_unload() / servable via vLLM LoRA —
                # no post-hoc Megatron-Bridge replay merger needed.
                # NOTE: only gated for LoRA runs — for full fine-tuning, 'hf_model'
                # would export the complete HF model (~440 GB for 235B) every save.
                "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]",
            ]
        )
    else:
        overrides.append("actor_rollout_ref.model.lora.rank=0")

    # Fused kernel overrides (gated by backend.use_fused_kernels)
    # The + prefix is Hydra syntax for appending keys not in the base config.
    # subprocess.run(cmd, ...) passes each arg directly — no shell quoting needed.
    if cfg.backend.use_fused_kernels:
        fk = cfg.backend.fused_kernels
        overrides.extend(
            [
                f"+actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion={fk.apply_rope_fusion}",
                f"+actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion={fk.masked_softmax_fusion}",
                f"+actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion={fk.bias_activation_fusion}",
                f"+actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion={fk.bias_dropout_fusion}",
                f"+actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion={fk.gradient_accumulation_fusion}",
                f"+actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs={fk.deallocate_pipeline_outputs}",
                f"+actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm={fk.persist_layer_norm}",
            ]
        )

    # Activation recomputation overrides (gated by model-specific config)
    # MoE models benefit from recomputation to reduce memory pressure.
    # These fields are optional in model YAMLs — models without them simply skip.
    recompute = OmegaConf.select(cfg.model, "recompute", default=None)
    if recompute is not None:
        overrides.extend(
            [
                f"+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method={recompute.method}",
                f"+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity={recompute.granularity}",
                f"+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers={recompute.num_layers}",
            ]
        )

    # MoE-specific override_transformer_config flags (gated by EP > 1).
    # These enable grouped GEMM, fused permutation, flex dispatcher, fp32 router,
    # DeepEP, and balanced pipeline split accounting for embedding/loss layers.
    # Applied to all MoE models (30B and 235B both benefit).
    if cfg.model.expert_parallel_size > 1:
        moe = cfg.backend.moe_overrides
        overrides.extend(
            [
                f"+actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm={moe.moe_grouped_gemm}",
                f"+actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion={moe.moe_permute_fusion}",
                f'+actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type="{moe.moe_token_dispatcher_type}"',
                f"+actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype={moe.moe_router_dtype}",
                f"+actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep={moe.moe_enable_deepep}",
                f"+actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split={moe.account_for_loss_in_pipeline_split}",
                f"+actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split={moe.account_for_embedding_in_pipeline_split}",
            ]
        )

    # Advanced optimizer offloading config (gated by model-specific config).
    # 235B needs CPU-offloaded optimizer with precision-aware updates and
    # overlapped D2H/H2D transfers.  Smaller models skip this entirely.
    optim_cfg = OmegaConf.select(cfg.model, "optimizer_config", default=None)
    if optim_cfg is not None:
        overrides.extend(
            [
                f"+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload={optim_cfg.optimizer_cpu_offload}",
                f"+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction={optim_cfg.optimizer_offload_fraction}",
                f"+actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d={optim_cfg.overlap_cpu_optimizer_d2h_h2d}",
                f"+actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer={optim_cfg.use_precision_aware_optimizer}",
            ]
        )

    return overrides


# =============================================================================
# Profiling overrides — backend-agnostic
# =============================================================================
def _build_profiling_overrides(cfg: DictConfig) -> list[str]:
    """Build verl overrides for profiling configuration."""
    p = cfg.profiling
    if not p.tool:
        return []

    save_path = f"{p.save_path}/{cfg.model.name}"
    # Format steps list for Hydra CLI: [3,4,5]
    steps_str = "[" + ",".join(str(s) for s in p.steps) + "]"

    overrides = [
        f"global_profiler.tool={p.tool}",
        f"global_profiler.steps={steps_str}",
        f"global_profiler.save_path={save_path}",
        f"global_profiler.profile_continuous_steps={p.profile_continuous_steps}",
        # Actor profiler (rollout inherits from actor via oc.select)
        f"actor_rollout_ref.actor.profiler.enable={p.actor_enable}",
        f"actor_rollout_ref.actor.profiler.all_ranks={p.actor_all_ranks}",
    ]

    if p.actor_ranks:
        ranks_str = "[" + ",".join(str(r) for r in p.actor_ranks) + "]"
        overrides.append(f"actor_rollout_ref.actor.profiler.ranks={ranks_str}")

    # Critic profiler (independent from actor)
    if p.critic_enable:
        overrides.extend(
            [
                f"critic.profiler.enable={p.critic_enable}",
                f"critic.profiler.all_ranks={p.critic_all_ranks}",
            ]
        )
        if p.critic_ranks:
            ranks_str = "[" + ",".join(str(r) for r in p.critic_ranks) + "]"
            overrides.append(f"critic.profiler.ranks={ranks_str}")

    return overrides


# =============================================================================
# Main override builder — dispatches by backend
# =============================================================================
def build_verl_overrides(cfg: DictConfig) -> list[str]:
    """Convert our Hydra config into verl CLI overrides."""
    overrides = _build_shared_overrides(cfg)

    if cfg.backend.name == "megatron":
        overrides.extend(_build_megatron_overrides(cfg))
    else:
        overrides.extend(_build_fsdp_overrides(cfg))

    # Profiling (backend-agnostic)
    overrides.extend(_build_profiling_overrides(cfg))

    # Logger (must be quoted for Hydra list parsing)
    logger_list = OmegaConf.to_container(cfg.tracking.logger)
    logger_str = '["' + '","'.join(logger_list) + '"]'
    overrides.append(f"trainer.logger={logger_str}")

    # Reward model — disable Reward Loop (verl 0.8.0.dev adds reward_model.enable
    # which defaults to False; explicit override for safety)
    overrides.append("reward.reward_model.enable=False")

    # Sandbox Fusion (conditional, backend-agnostic)
    # verl 0.8.0.dev moved custom_reward_function under the `reward` config group
    if cfg.sandbox.enabled:
        overrides.extend(
            [
                f"reward.custom_reward_function.path={cfg.compute.fsx_home}/custom_reward_fn.py",
                "reward.custom_reward_function.name=compute_score",
                f"+reward.custom_reward_function.reward_kwargs.sandbox_fusion_url={cfg.sandbox.url}",
                f"+reward.custom_reward_function.reward_kwargs.memory_limit_mb={cfg.sandbox.memory_limit_mb}",
            ]
        )

    # MLflow tracking URI (injected as Ray runtime env var)
    mlflow_uri = cfg.secrets.mlflow_tracking_uri
    if mlflow_uri:
        overrides.append(
            f"+ray_kwargs.ray_init.runtime_env.env_vars.MLFLOW_TRACKING_URI={mlflow_uri}"
        )

    # Cache directories — redirect from /tmp to /fsx to prevent disk exhaustion.
    # The Ray pip virtualenv + Triton/torch compile caches filled /tmp (100 GB)
    # on step 3 of the first FSDP run, causing an NCCL timeout (local notes).
    # NOTE: These MUST go through +ray_kwargs.ray_init.runtime_env.env_vars, not
    # runtime_env.yaml env_vars, because verl's inner ray.init() builds its own
    # runtime_env dict that overwrites the outer job-level env vars.
    # NOTE: Cache paths are STACK-VERSIONED.  Compile artifacts (Triton kernels,
    # torch.compile/Inductor autotune blocks, vLLM caches) are only valid for the
    # torch/vLLM/CUDA stack that produced them.  Sharing one cache dir across
    # image upgrades caused "InductorError: Failed to run autotuning code block:
    # CUDA driver error: file not found" when vLLM 0.20 hit artifacts compiled by
    # the old torch 2.9 / vLLM 0.12 stack.  Bump CACHE_STACK_VERSION whenever the
    # Docker base image (torch/vLLM/CUDA) changes.
    #
    # compile_cache=local moves them to node-local NVMe instead.  Required the
    # FIRST time a given max_model_len compiles: Triton writes a cubin then
    # immediately cuModuleLoads it, and on FSx with 8 ranks/node writing at once
    # the file is not yet visible -> the same "CUDA driver error: file not found"
    # from a completely different cause (write-visibility race, not staleness).
    # See the compile_cache block in conf/config.yaml for the full write-up.
    cache_mode = OmegaConf.select(cfg, "compile_cache", default="fsx")
    if cache_mode not in ("fsx", "local"):
        raise SystemExit(
            f"ABORT: invalid compile_cache={cache_mode!r}. Expected 'fsx' or 'local'."
        )
    if cache_mode == "local":
        cache_root = f"/tmp/compile-cache/{CACHE_STACK_VERSION}"
    else:
        cache_root = str(Path(cfg.compute.fsx_home).parent / "cache" / CACHE_STACK_VERSION)
    overrides.extend(
        [
            f"+ray_kwargs.ray_init.runtime_env.env_vars.TRITON_CACHE_DIR={cache_root}/triton",
            f"+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_COMPILE_CACHE_DIR={cache_root}/torch_compile",
            f"+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_CACHE_ROOT={cache_root}/vllm",
        ]
    )

    return overrides


def print_config_summary(cfg: DictConfig) -> None:
    """Print a human-readable configuration summary."""
    print("=" * 60)
    print("GRPO Training — Hydra Configuration")
    print("=" * 60)
    print(f"  Backend:   {cfg.backend.name}")
    print(f"  Model:     {cfg.model.name} ({cfg.model.fsx_path})")
    if cfg.lora.enabled:
        if cfg.backend.name == "megatron":
            lora_alpha = OmegaConf.select(
                cfg.model, "lora_alpha", default=cfg.lora.alpha
            )
            lora_targets = OmegaConf.select(
                cfg.model,
                "lora_target_modules",
                default=cfg.lora.megatron.target_modules,
            )
            lora_merge = OmegaConf.select(
                cfg.model, "lora_merge", default=cfg.lora.megatron.merge
            )
            merge_str = ", merge" if lora_merge else ""
            print(
                f"  LoRA:      rank={cfg.lora.rank}, alpha={lora_alpha}, "
                f"type={cfg.lora.megatron.type}{merge_str} (mbridge)"
            )
            print(f"  LoRA tgt:  {lora_targets}")
        else:
            print(
                f"  LoRA:      rank={cfg.lora.rank}, alpha={cfg.lora.alpha}, "
                f"targets={cfg.lora.target_modules} (PEFT)"
            )
    else:
        print("  LoRA:      DISABLED (full fine-tuning)")
    print(
        f"  Compute:   {cfg.compute.num_nodes} node(s) x {cfg.compute.gpus_per_node} GPU "
        f"({cfg.compute.num_nodes * cfg.compute.gpus_per_node} GPUs)"
    )
    if cfg.backend.name == "megatron":
        rollout_tp = OmegaConf.select(
            cfg.model,
            "rollout_tensor_parallel_size",
            default=cfg.model.tensor_parallel_size,
        )
        parallel_str = (
            f"  Parallel:  TP={cfg.model.tensor_parallel_size}, "
            f"PP={cfg.model.pipeline_parallel_size}, "
            f"EP={cfg.model.expert_parallel_size}, "
            f"CP={cfg.model.context_parallel_size}"
        )
        if rollout_tp != cfg.model.tensor_parallel_size:
            parallel_str += f", Rollout TP={rollout_tp}"
        print(parallel_str)
    else:
        print(f"  TP:        {cfg.model.tensor_parallel_size}")
    print(
        f"  Batch:     {cfg.training.train_batch_size} "
        f"(n={cfg.training.n_responses_per_prompt} per prompt)"
    )
    # Show effective mini-batch size when a per-model override is active
    effective_mbs = OmegaConf.select(cfg.model, "ppo_mini_batch_size", default=None)
    if effective_mbs is not None:
        print(
            f"  MiniBatch: {effective_mbs} (model override, "
            f"default={cfg.training.ppo_mini_batch_size})"
        )
    print(f"  LR:        {cfg.training.learning_rate}")
    print(f"  Epochs:    {cfg.training.total_epochs}")
    logger_list = OmegaConf.to_container(cfg.tracking.logger)
    print(f"  Tracking:  {', '.join(logger_list)}")
    if cfg.sandbox.enabled:
        print(f"  Sandbox:   {cfg.sandbox.url}")
    else:
        print("  Sandbox:   DISABLED (local execution)")
    if cfg.profiling.tool:
        print(f"  Profiling: {cfg.profiling.tool} (steps {list(cfg.profiling.steps)})")
    else:
        print("  Profiling: disabled")
    print(f"  Ray:       {cfg.ray.address}")
    print("=" * 60)


def preflight_sandbox_check(cfg: DictConfig) -> None:
    """Abort submission if the code-execution sandbox is unhealthy.

    Runs a tiny `print(42)` through the sandbox's /run_code endpoint from a
    pod inside the cluster (the sandbox URL is cluster-internal DNS, not
    reachable from the submit host). Guards against the OTel-injection
    meltdown where sandbox latency ballooned ~0.03s -> ~10s and every code
    reward silently zeroed out, poisoning training. No-op when the sandbox is
    disabled or preflight is turned off.
    """
    if not cfg.sandbox.enabled:
        return
    pf = OmegaConf.select(cfg.sandbox, "preflight", default=None)
    if pf is None or not pf.get("enabled", False):
        return

    url = cfg.sandbox.url
    max_latency = float(pf.max_latency_s)
    req_timeout = int(pf.request_timeout_s)
    pod = pf.exec_pod
    namespace = pf.namespace
    context = OmegaConf.select(pf, "kube_context", default=None)

    print("\n" + "=" * 60)
    print("SANDBOX PREFLIGHT CHECK")
    print(f"  Probing {url}")
    print(f"  via pod {namespace}/{pod}  (max latency {max_latency}s)")

    # Probe script runs *inside* the cluster pod. Prints one JSON line.
    # url/timeout are injected as literal reprs at the top so the body has no
    # format placeholders (keeps the code static and ruff-clean).
    probe = (
        f"u={url!r}\n"
        f"to={req_timeout}\n"
        "import json,time,urllib.request\n"
        "t=time.time()\n"
        "try:\n"
        "  req=urllib.request.Request(u,data=json.dumps({'code':'print(42)','language':'python'}).encode(),"
        "headers={'Content-Type':'application/json'})\n"
        "  r=json.loads(urllib.request.urlopen(req,timeout=to).read())\n"
        "  lat=time.time()-t\n"
        "  rr=r.get('run_result') or {}\n"
        "  out=(rr.get('stdout') or '').strip()\n"
        "  print(json.dumps({'ok':r.get('status')=='Success' and out=='42','status':r.get('status'),"
        "'latency':round(lat,3),'stdout':out}))\n"
        "except Exception as e:\n"
        "  print(json.dumps({'ok':False,'error':str(e),'latency':round(time.time()-t,3)}))\n"
    )

    kexec = ["kubectl"]
    if context:
        kexec += ["--context", context]
    kexec += ["exec", "-n", namespace, pod, "--", "python3", "-c", probe]

    try:
        proc = subprocess.run(
            kexec, capture_output=True, text=True, timeout=req_timeout + 30, check=False
        )
    except subprocess.TimeoutExpired:
        print("  RESULT: FAILED — kubectl exec timed out")
        print("=" * 60)
        sys.exit(
            "ABORT: sandbox preflight timed out. Sandbox is likely wedged. "
            f"Check pods: kubectl get pods -n {namespace} -l app=sandbox-fusion. "
            "Override with sandbox.preflight.enabled=false to bypass."
        )

    # The probe prints one JSON line to stdout; kubectl may prepend warnings.
    result = None
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                result = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if result is None:
        print("  RESULT: FAILED — could not parse probe output")
        print(f"  stdout: {proc.stdout[-500:]}")
        print(f"  stderr: {proc.stderr[-500:]}")
        print("=" * 60)
        sys.exit(
            "ABORT: sandbox preflight produced no result (kubectl/pod issue). "
            "Override with sandbox.preflight.enabled=false to bypass."
        )

    lat = result.get("latency", -1)
    if not result.get("ok"):
        print(f"  RESULT: FAILED — {result}")
        print("=" * 60)
        sys.exit(
            f"ABORT: sandbox is unhealthy (status={result.get('status')}, "
            f"latency={lat}s, err={result.get('error')}). Code rewards would "
            "zero out and poison training. Fix the sandbox (check OTel injection "
            "+ pod health) before submitting. Override with "
            "sandbox.preflight.enabled=false to bypass."
        )

    if lat > max_latency:
        print(f"  RESULT: FAILED — latency {lat}s exceeds max {max_latency}s")
        print("=" * 60)
        sys.exit(
            f"ABORT: sandbox latency {lat}s exceeds max {max_latency}s. Sandbox "
            "is degraded (likely OTel-injection overhead) and code scoring will "
            "time out. Fix before submitting. Override with "
            "sandbox.preflight.enabled=false."
        )

    print(f"  RESULT: OK — status=Success, latency={lat}s")
    print("=" * 60)


def preflight_resume_check(cfg: DictConfig) -> None:
    """Make the resume target explicit before submitting.

    `resume_mode=auto` (the default) silently resumes from whatever step the
    tracker file in default_local_dir points at. That is what you want after a
    crash, but it is also how a run gets restarted on top of a checkpoint that
    was trained against a broken reward signal without anyone noticing — which
    is exactly what happened when a run resumed from a poisoned global_step_350
    and never recovered. Print the resolved resume target so the decision shows
    up in the submit log instead of being implicit.
    """
    mode = OmegaConf.select(cfg.backend, "resume_mode", default="auto")
    ckpt_dir = f"{cfg.compute.fsx_home}/ckpts/{cfg.data_name}/{cfg.backend.name}/{cfg.model.name}"

    print("\n" + "=" * 60)
    print("RESUME PREFLIGHT")
    print(f"  resume_mode:       {mode}")
    print(f"  default_local_dir: {ckpt_dir}")

    # verl v0.8.0 (ray_trainer.py::_load_checkpoint) only recognises these three.
    # An unrecognised value does NOT raise there — it silently falls through to
    # "newest checkpoint in default_local_dir", so a typo'd mode looks like it
    # honoured resume_from_path while actually resuming from somewhere else (or
    # crashing with 'NoneType' has no attribute 'split' on an empty tree).
    # Fail loudly here instead.
    valid = ("auto", "disable", "resume_path")
    if mode not in valid:
        print("=" * 60)
        sys.exit(
            f"ABORT: invalid backend.resume_mode={mode!r}. verl accepts only {valid}. "
            "NOTE: the explicit-path mode is 'resume_path' (NOT 'resume_from_path' — that is "
            "the name of the *path* field). An invalid value is silently ignored by verl and "
            "resumes from the newest checkpoint instead."
        )

    if mode == "resume_path":
        path = OmegaConf.select(cfg.backend, "resume_from_path", default=None)
        if not path:
            print("=" * 60)
            sys.exit(
                "ABORT: resume_mode=resume_path but backend.resume_from_path is unset. "
                "Set backend.resume_from_path=<checkpoint dir> or use resume_mode=auto."
            )
        if "global_step_" not in str(path):
            print("=" * 60)
            sys.exit(f"ABORT: backend.resume_from_path must contain 'global_step_': {path}")
        print(f"  --> resuming from EXPLICIT path:\n        {path}")
        print("=" * 60)
        return

    if mode == "disable":
        print("  --> fresh start (checkpoint resume disabled)")
        print("=" * 60)
        return

    # resume_mode=auto — resolve the tracker file so the implicit target is shown.
    pf = OmegaConf.select(cfg.sandbox, "preflight", default=None)
    if pf is None:
        print("  --> WARNING: resume_mode=auto and the resume target could not be resolved.")
        print("      Training will resume from the newest checkpoint in default_local_dir.")
        print("=" * 60)
        return

    tracker = f"{ckpt_dir}/latest_checkpointed_iteration.txt"
    kexec = ["kubectl"]
    context = OmegaConf.select(pf, "kube_context", default=None)
    if context:
        kexec += ["--context", context]
    kexec += ["exec", "-n", pf.namespace, pf.exec_pod, "--", "sh", "-c", f"cat {tracker} 2>/dev/null || true"]

    step = ""
    try:
        proc = subprocess.run(kexec, capture_output=True, text=True, timeout=60, check=False)
        step = proc.stdout.strip()
    except subprocess.TimeoutExpired:
        step = ""

    if step:
        print(f"  --> WARNING: will SILENTLY RESUME from global_step_{step}")
        print(f"      (tracker: {tracker})")
        print("      If that checkpoint is not known-good, abort now and pass:")
        print("        backend.resume_mode=resume_path \\")
        print(f"        backend.resume_from_path={ckpt_dir}/global_step_<N>")
    else:
        print("  --> no tracker file found: starting from the base model (fresh run)")
    print("=" * 60)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Resolve all interpolations (oc.env, cross-references)
    OmegaConf.resolve(cfg)

    # Print summary
    print_config_summary(cfg)

    # Pre-submit sandbox health guardrail (aborts on failure)
    preflight_sandbox_check(cfg)

    # Make the resume target explicit (guards against silently resuming onto a
    # checkpoint trained against a broken reward signal)
    preflight_resume_check(cfg)

    # Build verl CLI overrides
    verl_overrides = build_verl_overrides(cfg)

    # Build the ray job submit command
    script_dir = Path(__file__).parent
    runtime_env_path = script_dir / "runtime_env.yaml"

    cmd = [
        "ray",
        "job",
        "submit",
        "--address",
        cfg.ray.address,
    ]
    if cfg.ray.headers:
        cmd.extend(["--headers", cfg.ray.headers])
    if cfg.ray.no_wait:
        cmd.append("--no-wait")
    cmd.extend(
        [
            "--runtime-env",
            str(runtime_env_path),
            "--",
            "python3",
            "-m",
            "verl.trainer.main_ppo",
        ]
    )

    # Megatron requires a different verl base config
    if cfg.backend.verl_config_name:
        cmd.extend(
            [
                "--config-path=config",
                f"--config-name={cfg.backend.verl_config_name}",
            ]
        )

    cmd.extend(verl_overrides)

    # Print the command for reproducibility
    print("\nRay job command:")
    entrypoint_start = cmd.index("--") + 1
    entrypoint = cmd[entrypoint_start:]
    # Print entrypoint header (python3 -m verl.trainer.main_ppo [--config-*])
    header_parts = ["python3", "-m", "verl.trainer.main_ppo"]
    override_start = 3
    if cfg.backend.verl_config_name:
        header_parts.extend(
            [
                "--config-path=config",
                f"--config-name={cfg.backend.verl_config_name}",
            ]
        )
        override_start = 5
    print(f"  {' '.join(cmd[:entrypoint_start])} \\")
    print(f"    {' '.join(header_parts)} \\")
    for override in entrypoint[override_start:]:
        print(f"    {override} \\")
    print()

    # Submit
    print("Submitting Ray job...")
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
