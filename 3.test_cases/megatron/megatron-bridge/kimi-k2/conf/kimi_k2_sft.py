# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Megatron-Bridge full-parameter SFT config for Kimi K2 (1.04T MoE) on 256x B300.

This module builds and returns a Megatron-Bridge ``ConfigContainer`` for supervised
fine-tuning (full parameter, BF16) of Moonshot AI's Kimi K2 on an EKS cluster of
32x p6-b300.48xlarge (256x NVIDIA B300). The MoE token dispatch is routed through the
``flex``/``deepep`` backend so that, at runtime, Megatron-Core's DeepEP dispatcher
resolves ``import deep_ep`` to UCCL's EFA implementation (the ``deep_ep`` shadow module
installed from ``/opt/uccl/ep/deep_ep_wrapper``), giving expert-parallel all-to-all over
AWS EFA instead of NVSHMEM/IB-bound NVIDIA DeepEP.

Usage (inside the container, launched by torchrun on all 256 ranks):

    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.gpt_step import forward_step      # SFT forward/backward
    from conf.kimi_k2_sft import kimi_k2_sft_config

    finetune(config=kimi_k2_sft_config(), forward_step_func=forward_step)

Grounded against Megatron-Bridge ``main`` (image ships v0.4.0):
- recipe skeleton  : src/megatron/bridge/recipes/common.py :: _sft_common()
- SFT recipe shape : src/megatron/bridge/recipes/llama/llama3.py :: llama3_70b_sft_config()
- MoE/MTP fields   : src/megatron/bridge/recipes/deepseek/deepseek_v3.py
- finetune entry   : src/megatron/bridge/training/finetune.py :: finetune(config, forward_step_func)
  https://github.com/NVIDIA-NeMo/Megatron-Bridge
"""

import logging
import os

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.training.config import ConfigContainer

logger = logging.getLogger(__name__)

# TODO(validate against image): apply_flex_dispatcher_backend module path/symbol
# unverified against Bridge v0.4.0 — the moe_token_dispatcher_type/moe_flex_dispatcher_backend
# fields may suffice. Config import must NOT crash if the symbol is absent.
# https://github.com/NVIDIA-NeMo/Megatron-Bridge
try:
    from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
except (ImportError, AttributeError) as _e:
    apply_flex_dispatcher_backend = None
    logger.warning(
        "apply_flex_dispatcher_backend unavailable (%s); relying on "
        "moe_token_dispatcher_type/moe_flex_dispatcher_backend fields instead.",
        _e,
    )

# ---------------------------------------------------------------------------
# Paths on the shared FSx for Lustre filesystem (mounted at /fsx in the PyTorchJob).
# Override via env vars so the manifest can point at a specific run directory
# without editing this file.
# ---------------------------------------------------------------------------
FSX_ROOT = os.environ.get("FSX_ROOT", "/fsx/kimi-k2")

# Megatron-Core distributed checkpoint produced by the HF -> MCore conversion step
# (Megatron-Bridge AutoBridge.import_ckpt; see 2.*.sh / README). BF16, ~2 TB.
PRETRAINED_CHECKPOINT = os.environ.get(
    "KIMI_K2_MCORE_CKPT", os.path.join(FSX_ROOT, "mcore")
)
# SFT training output (saved torch_dist checkpoints + resume).
OUTPUT_DIR = os.environ.get("KIMI_K2_OUTPUT_DIR", os.path.join(FSX_ROOT, "sft-output"))
# Tokenizer: the HF Kimi K2 repo dir on FSx (config + tokenizer). We pass a local
# path so no network/auth is needed on the workers.
HF_MODEL_PATH = os.environ.get("KIMI_K2_HF_PATH", os.path.join(FSX_ROOT, "hf"))

# HF model id for architecture/config resolution. Full SFT starts from the *Base* repo.
# block-FP8 weights are dequantized to BF16 during the offline conversion step; here we
# only need the architecture (load_weights=False), so either the HF id or the local
# HF_MODEL_PATH works as the source for the model provider.
KIMI_K2_HF_ID = os.environ.get("KIMI_K2_HF_ID", "moonshotai/Kimi-K2-Base")

# ---------------------------------------------------------------------------
# Parallelism (256 GPUs), read from env so the manifest is the single source of
# truth. Defaults are the canonical layout: TP*PP*DP = 8*8*4 = 256 (world size);
# EP=32 spans TP*DP = 32; 384 routed experts / 32 = 12 experts per EP rank.
# ---------------------------------------------------------------------------
TENSOR_PARALLEL = int(os.environ.get("TENSOR_PARALLEL", "8"))
EXPERT_PARALLEL = int(os.environ.get("EXPERT_PARALLEL", "32"))
PIPELINE_PARALLEL = int(os.environ.get("PIPELINE_PARALLEL", "8"))
DATA_PARALLEL = int(os.environ.get("DATA_PARALLEL", "4"))


def kimi_k2_sft_config() -> ConfigContainer:
    """Return a full-parameter SFT ``ConfigContainer`` for Kimi K2 on 256x B300.

    Parallelism (256 GPUs, expert_tensor_parallel_size=1):
        TP=8, PP=8, DP=4  ->  world = TP*PP*DP = 8*8*4 = 256.
        DP = world/(TP*PP) = 256/64 = 4.
        EP=32             ->  EP spans TP*DP = 32; 384 routed experts / 32 = 12 per EP rank.
                              EP must divide TP*DP = 32 with ETP=1 -> 32 | 32 OK.
        CP=1.
    Memory sanity: ~16 B/param (BF16 weights+grads + FP32 Adam moments, distributed
    optimizer) x 1.04T / 256 ~= 65 GB/GPU of sharded model state, leaving headroom on
    288 GB B300 HBM for activations. Activations use full recompute when
    MOE_A2A_OVERLAP=off; overlap=on requires recompute OFF + VPP=2 on core 0.17.1
    (higher activation memory — exercised empirically by the kimi-k2 benchmark cells).
    """
    # Base SFT template: lower LR (5e-6), cosine schedule, bf16_mixed, packed sequences,
    # SQuAD default dataset (overridden below), torch_dist checkpoints, seed 5678.
    # NOTE: _sft_common is a private (underscore) Bridge API and may move between
    # versions; pinned image is Bridge v0.4.0.
    cfg = _sft_common()

    # -- Model provider from the Kimi K2 architecture ------------------------
    # AutoBridge detects the HF architecture and returns a Megatron-Core provider.
    # load_weights=False: real weights are loaded from the converted MCore checkpoint
    # via cfg.checkpoint.pretrained_checkpoint (set below), not re-converted here.
    # Validated on the image (Bridge 0.4.2, nemo:26.04.01): moonshotai/Kimi-K2-Base
    # declares architectures=["DeepseekV3ForCausalLM"], which AutoBridge routes to the
    # registered DeepSeekV3Bridge, building the literal Kimi K2 provider (384 experts,
    # moe_router_num_groups=1, 64 heads, MLA) straight from the HF config.
    # trust_remote_code=True is REQUIRED: the HF config uses auto_map ->
    # configuration_deepseek.DeepseekV3Config (custom code); without it the config load
    # fails with "contains custom code which must be executed".
    cfg.model = AutoBridge.from_hf_pretrained(
        KIMI_K2_HF_ID, trust_remote_code=True
    ).to_megatron_provider(load_weights=False)

    # Tokenizer from the local HF repo dir on FSx (no network/auth on workers).
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = HF_MODEL_PATH

    # -- Sequence length -----------------------------------------------------
    seq_length = 4096
    cfg.model.seq_length = seq_length
    cfg.dataset.seq_length = seq_length
    # Packed-sequence size must match seq_length (mirrors the llama SFT recipes).
    cfg.dataset.packed_sequence_specs.packed_sequence_size = seq_length
    if cfg.model.context_parallel_size and cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # -- Parallelism (256 GPUs) ---------------------------------------------
    # DP = world/(TP*PP) is derived by Megatron from the launcher world size; the
    # DATA_PARALLEL env (default 4) is the manifest-side contract for that value.
    cfg.model.tensor_model_parallel_size = TENSOR_PARALLEL    # intra-node NVLink
    cfg.model.pipeline_model_parallel_size = PIPELINE_PARALLEL  # spans nodes; fits ~61 layers
    cfg.model.expert_model_parallel_size = EXPERT_PARALLEL    # 384 % 32 == 0
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True                # required with TP>1 for MoE/MLA
    cfg.model.pipeline_dtype = torch.bfloat16
    # Kimi K2 ships NO multi-token-prediction layer (HF num_nextn_predict_layers=0). Set it
    # explicitly to None — NOT 0: the DSV3 pipeline-layout helper treats None as no-MTP
    # (`None or 0` -> ["loss"] tail; an ABSENT attr would default to 1 and corrupt the
    # layout), and core's comm-overlap setup asserts `mtp_num_layers is None or == 1` when
    # overlap_moe_expert_parallel_comm is enabled — an int 0 trips that assert.
    cfg.model.mtp_num_layers = None
    # NOTE: virtual_pipeline_model_parallel_size and the explicit 61-layer pipeline layout
    # are finalized in the "Recompute + VPP + pipeline layout" section below — on
    # Megatron-Core 0.17.1, overlap=on requires VPP + recompute OFF, and the layout
    # depends on (PP, VPP), so the three are set together.

    # -- MoE token dispatcher: A/B toggle (see ../../dsv3/) --------------------
    # MOE_DISPATCHER selects the expert-parallel all-to-all backend so the benchmark
    # harness can compare them with everything else held fixed:
    #   "deepep"   -> flex dispatcher + UCCL's `deep_ep` shadow over EFA (treatment, default)
    #   "alltoall" -> Megatron-Core NCCL all-to-all over EFA (baseline, "without DeepEP")
    # The flex dispatcher decouples the EP comm group from model parallelism; the "deepep"
    # backend makes Megatron-Core call into the top-level `deep_ep` python module, which on
    # this image is UCCL's EFA drop-in (installed from /opt/uccl/ep/deep_ep_wrapper).
    # See ../../dsv3/README.md for the A/B methodology.
    moe_dispatcher = os.environ.get("MOE_DISPATCHER", "deepep").lower()
    if moe_dispatcher == "alltoall":
        cfg.model.moe_token_dispatcher_type = "alltoall"
    elif moe_dispatcher == "deepep":
        cfg.model.moe_token_dispatcher_type = "flex"
        cfg.model.moe_flex_dispatcher_backend = "deepep"
        # TODO(validate against image): canonical recipe also calls for
        # cfg.model.moe_enable_deepep = True. This field name is unverified against Bridge
        # v0.4.0 / MCore on the image (the recipes use moe_token_dispatcher_type +
        # moe_flex_dispatcher_backend + apply_flex_dispatcher_backend instead). Set it only
        # if the MCore TransformerConfig on the image exposes it.
        # docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/moe.html
        if hasattr(cfg.model, "moe_enable_deepep"):
            cfg.model.moe_enable_deepep = True
    else:
        raise ValueError(
            "MOE_DISPATCHER must be 'deepep' or 'alltoall', got %r" % moe_dispatcher
        )
    # MoE perf knobs grafted from the DeepSeek-V3 recipe (safe defaults for fine-grained MoE).
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_shared_expert_overlap = True
    cfg.model.moe_router_force_load_balancing = False
    # apply_flex_dispatcher_backend finalizes the flex/deepep wiring (deepseek recipe calls
    # it last). Only relevant for the flex dispatcher; skip it for the alltoall baseline.
    # Guarded: the symbol may be absent on the image (see import above) — fall back to the
    # moe_token_dispatcher_type/moe_flex_dispatcher_backend fields already set.
    if moe_dispatcher == "deepep" and apply_flex_dispatcher_backend is not None:
        try:
            apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)
        except (ImportError, AttributeError) as _e:
            logger.warning(
                "apply_flex_dispatcher_backend call failed (%s); relying on "
                "moe_token_dispatcher_type/moe_flex_dispatcher_backend fields instead.",
                _e,
            )

    # -- A2A / EP communication overlap (decisive A/B confounder) --------------
    # MOE_A2A_OVERLAP in {on, off}, held IDENTICAL across both dispatcher arms within a
    # run. "on" overlaps the expert-parallel all-to-all with compute (realistic deployment;
    # Megatron 1F1B hides up to ~93% of A2A) -> report as the deployment delta. "off"
    # exposes the A2A fully (dispatcher-isolation upper bound). See ../../dsv3/README.md.
    moe_a2a_overlap = os.environ.get("MOE_A2A_OVERLAP", "on").lower() == "on"
    # Validated on the image (core 0.17.1, 256x B300 A/B): set the flag on whichever config
    # object exposes it — the model config is what the validator reads; comm_overlap mirrors
    # it. delay_wgrad_compute is held OFF to minimise the constraint surface; enabling it is
    # an unvalidated follow-up (see ../../dsv3/benchmarks/RESULTS.md).
    for _obj in (getattr(cfg, "comm_overlap", None), cfg.model):
        if _obj is None:
            continue
        if hasattr(_obj, "overlap_moe_expert_parallel_comm"):
            _obj.overlap_moe_expert_parallel_comm = moe_a2a_overlap
        if hasattr(_obj, "delay_wgrad_compute"):
            _obj.delay_wgrad_compute = False

    # -- Precision / optimizer ----------------------------------------------
    cfg.mixed_precision = "bf16_mixed"
    # Distributed (ZeRO-style) optimizer is mandatory at this scale; the SFT template
    # defaults it False, so enable it explicitly. FP32 Adam moments for SFT stability.
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.adam_beta2 = 0.98

    # -- Recompute + VPP + pipeline layout (coupled; validated on core 0.17.1) ----
    # overlap=on hard-requires a virtual pipeline (when PP>1) AND recompute fully OFF;
    # the 61-layer pipeline layout depends on (PP, VPP). Finalize all three together,
    # mirroring the validated benchmark entrypoint (../benchmarks/bench_kimi_k2_pretrain.py).
    if moe_a2a_overlap and PIPELINE_PARALLEL > 1:
        cfg.model.virtual_pipeline_model_parallel_size = 2   # recipe's (8,2) 16-chunk layout
        cfg.model.recompute_granularity = None
        cfg.model.recompute_method = None
        cfg.model.recompute_num_layers = None
        cfg.model.recompute_modules = None
    else:
        cfg.model.virtual_pipeline_model_parallel_size = None
        cfg.model.recompute_granularity = "full"             # fit activation memory at scale
        cfg.model.recompute_method = "uniform"
        cfg.model.recompute_num_layers = 1
        cfg.model.recompute_modules = None
    # Kimi K2's 61 layers do NOT divide evenly over PP=8 — an explicit layout is REQUIRED
    # (None errors at model build). Reuse the DSV3 recipe helper: it is MTP-aware (mtp=0 set
    # above -> last stage ["loss"], no "mtp") and ships layouts for (pp,vpp) in
    # {(4,1),(8,1),(16,1),(4,2),(8,2),(4,4)}. Validated on the image (Bridge 0.4.2) by the
    # kimi-k2 config-assembly gate.
    from megatron.bridge.recipes.deepseek.deepseek_v3 import (
        set_deepseek_v3_pipeline_model_parallel_layout,
    )
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
    if (cfg.model.pipeline_model_parallel_size > 1
            and cfg.model.pipeline_model_parallel_layout is None):
        raise ValueError(
            "no shipped 61-layer pipeline layout for (pp=%s, vpp=%s); use PP in {4,8,16} "
            "(see megatron.bridge.recipes.deepseek.deepseek_v3."
            "set_deepseek_v3_pipeline_model_parallel_layout)"
            % (cfg.model.pipeline_model_parallel_size,
               cfg.model.virtual_pipeline_model_parallel_size or 1)
        )
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"  # CUDA graphs + EP all-to-all do not mix well

    # -- SFT hyperparameters -------------------------------------------------
    # TRAIN_ITERS is env-overridable so the benchmark harness (../../dsv3/) can run a
    # short A/B (e.g. WARMUP_ITERS+MEASURE_ITERS) without editing this file; the full
    # SFT default is 2000.
    cfg.train.train_iters = int(os.environ.get("TRAIN_ITERS", "2000"))
    cfg.train.global_batch_size = 256   # 256 = DP(4) * grad-accum; tune to throughput
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 100
    cfg.validation.eval_iters = 32
    cfg.scheduler.lr_warmup_iters = 100
    cfg.scheduler.max_lr = 5e-6         # low LR for full-parameter SFT
    cfg.scheduler.min_lr = 0.0
    cfg.logger.log_interval = 1

    # -- Dataset on FSx ------------------------------------------------------
    # _sft_common defaults to default_squad_config (HF SQuAD). Point at the prepared
    # SFT dataset on FSx instead.
    # TODO(validate against image): confirm the dataset config field that selects a
    # local SFT corpus on Bridge v0.4.0. The FinetuningDatasetConfig / HFDatasetConfig
    # produced by default_squad_config exposes a dataset root/name; set the on-FSx
    # path here (e.g. cfg.dataset.dataset_root = SFT_DATASET_DIR, or build a
    # FinetuningDatasetConfig pointing at the jsonl). Do not invent the field name.
    # https://docs.nvidia.com/nemo/megatron-bridge/0.4.0/apidocs/bridge/bridge.recipes.utils.dataset_utils.html
    SFT_DATASET_DIR = os.environ.get("KIMI_K2_SFT_DATA", os.path.join(FSX_ROOT, "sft-data"))
    if hasattr(cfg.dataset, "dataset_root"):
        cfg.dataset.dataset_root = SFT_DATASET_DIR

    # -- Checkpoint / IO on FSx ---------------------------------------------
    cfg.checkpoint.pretrained_checkpoint = PRETRAINED_CHECKPOINT  # converted MCore (BF16)
    cfg.checkpoint.save = os.path.join(OUTPUT_DIR, "checkpoints")
    cfg.checkpoint.load = os.path.join(OUTPUT_DIR, "checkpoints")  # resume from latest
    cfg.checkpoint.save_interval = 200
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True
    cfg.checkpoint.async_save = True

    cfg.logger.tensorboard_dir = os.path.join(OUTPUT_DIR, "tb_logs")

    return cfg


if __name__ == "__main__":
    # Launched by torchrun on all 256 ranks (see the PyTorchJob manifest). Build the
    # config and hand it to Megatron-Bridge's finetune entry point.
    cfg = kimi_k2_sft_config()
    print("Kimi K2 SFT ConfigContainer built.")
    print(
        f"  TP={cfg.model.tensor_model_parallel_size} "
        f"PP={cfg.model.pipeline_model_parallel_size} "
        f"EP={cfg.model.expert_model_parallel_size} "
        f"ETP={cfg.model.expert_tensor_parallel_size} "
        f"CP={cfg.model.context_parallel_size}"
    )

    # TODO(validate against image): the finetune entry point and forward_step_func
    # symbol are unverified against Bridge v0.4.0. The shape below mirrors the
    # Bridge SFT recipes (training/finetune.py::finetune and training/gpt_step.py::
    # forward_step); confirm both against the image before a real run.
    # https://github.com/NVIDIA-NeMo/Megatron-Bridge
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.gpt_step import forward_step

    finetune(config=cfg, forward_step_func=forward_step)
