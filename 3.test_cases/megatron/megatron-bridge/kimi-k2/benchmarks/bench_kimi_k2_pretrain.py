# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Throughput + loss-equivalence A/B entrypoint: Kimi-K2 384-expert MoE pretrain step.

Builds the LITERAL Kimi-K2 architecture (384 routed experts, 64 attention heads, node-group
routing n_group=1, MLA, 61 layers, NO MTP) via Megatron-Bridge's AutoBridge from the HF config,
with **mock data** and **random-init weights**, then runs ``pretrain()`` for a fixed number of
iterations. The ONLY thing that changes between the two benchmark arms is the MoE token
dispatcher, selected by ``MOE_DISPATCHER``:

    MOE_DISPATCHER=alltoall  -> moe_token_dispatcher_type="alltoall"  (NCCL all-to-all / EFA)  [baseline]
    MOE_DISPATCHER=deepep    -> flex + moe_flex_dispatcher_backend="deepep" (UCCL EFA drop-in) [treatment]

This mirrors ``../../dsv3/benchmarks/bench_dsv3_pretrain.py`` but swaps the recipe-native DeepSeek-V3
256-expert model for the real Kimi-K2 provider:

- the DSV3 recipe ``deepseek_v3_pretrain_config_32nodes()`` supplies the model-agnostic
  scaffolding (MOCK dataset, training/optimizer/ddp/logger config, seed);
- ``AutoBridge.from_hf_pretrained(<hf>, trust_remote_code=True).to_megatron_provider(load_weights=False)``
  supplies the correct Kimi-K2 model provider (all coupled MLA / expert / group dims read from
  the HF config); we graft it onto ``cfg.model`` and re-derive the 61-layer pipeline layout.

Validated this session on the image (Bridge 0.4.2, nemo:26.04.01): AutoBridge routes
Kimi-K2-Base (``architectures=["DeepseekV3ForCausalLM"]``) to DeepSeekV3Bridge and yields
num_moe_experts=384, moe_router_num_groups=1, num_attention_heads=64, num_layers=61,
multi_latent_attention=True, mtp_num_layers=0. ``trust_remote_code=True`` is REQUIRED (the HF
config uses auto_map -> configuration_deepseek.DeepseekV3Config, custom code).

Why this is a valid A/B: model/data/parallelism/precision/seed are byte-identical across arms,
so the iter-time ratio isolates the dispatcher. The per-iteration ``lm loss`` line (log_interval=1)
is compared across the two arms to confirm they perform the same numerical work (a dispatcher
that dropped/mis-routed tokens would diverge). All knobs come from env so the launcher is the
single source of truth and both arms differ only in MOE_DISPATCHER (and MOE_A2A_OVERLAP, held
identical across arms within a run).
"""

import logging
import os

import torch

logger = logging.getLogger("bench_kimi_k2_pretrain")
logging.basicConfig(level=logging.INFO)

# HF Kimi-K2 repo dir staged on FSx (config.json + configuration_deepseek.py; weights unused).
HF_PATH = os.environ.get("KIMI_K2_HF_PATH", "/fsx/kimi-k2/hf")


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def build_config():
    from megatron.bridge import AutoBridge
    from megatron.bridge.recipes.deepseek.deepseek_v3 import (
        deepseek_v3_pretrain_config_32nodes,
        apply_flex_dispatcher_backend,
        set_deepseek_v3_pipeline_model_parallel_layout,
    )

    # Parallelism — manifest/launcher contract. Canonical 256-GPU layout:
    # TP8 * PP8 = 64; world 256 -> DP=4; EP=32 divides TP*DP=32 (ETP=1). 384 experts / 32 = 12/rank.
    tp = _int("TENSOR_PARALLEL", 8)
    pp = _int("PIPELINE_PARALLEL", 8)
    ep = _int("EXPERT_PARALLEL", 32)
    cp = _int("CONTEXT_PARALLEL", 1)

    train_iters = _int("TRAIN_ITERS", 24)
    global_batch = _int("GLOBAL_BATCH", 256)
    micro_batch = _int("MICRO_BATCH", 1)
    seq_len = _int("SEQ_LEN", 4096)

    # 1) DSV3 recipe supplies the model-agnostic scaffolding (mock data by default via
    #    cfg.dataset.blend=None, plus train/optim/ddp/logger/seed). We keep all of that and
    #    replace ONLY cfg.model with Kimi-K2.
    cfg = deepseek_v3_pretrain_config_32nodes()

    # 2) Build the literal Kimi-K2 provider from the HF config (random init; no ~2 TB weights).
    #    trust_remote_code=True is REQUIRED (config auto_map -> configuration_deepseek.DeepseekV3Config).
    k2 = AutoBridge.from_hf_pretrained(HF_PATH, trust_remote_code=True).to_megatron_provider(
        load_weights=False
    )
    cfg.model = k2
    m = cfg.model

    # 3) Re-apply the runtime knobs. The AutoBridge provider carries only the architecture; the
    #    recipe's runtime/parallelism settings lived on the model object we just replaced, so we
    #    set them explicitly here (mirrors conf/kimi_k2_sft.py + dsv3/benchmarks/bench_dsv3_pretrain.py).
    m.tensor_model_parallel_size = tp
    m.pipeline_model_parallel_size = pp
    m.expert_model_parallel_size = ep
    m.expert_tensor_parallel_size = 1
    m.context_parallel_size = cp
    m.sequence_parallel = tp > 1
    m.seq_length = seq_len
    m.pipeline_dtype = torch.bfloat16
    m.transformer_impl = "transformer_engine"
    if hasattr(m, "cuda_graph_impl"):
        m.cuda_graph_impl = "none"          # CUDA graphs + EP all-to-all do not mix well
    m.moe_grouped_gemm = True
    m.moe_permute_fusion = True
    # Kimi-K2 has NO multi-token-prediction layer (DSV3 ships MTP=1). Use None — NOT 0:
    # the layout helper treats None as no-MTP (`None or 0` -> ["loss"] tail), and core's
    # comm-overlap setup asserts `mtp_num_layers is None or == 1` when
    # overlap_moe_expert_parallel_comm is enabled — an int 0 trips that assert
    # ("MTP layernum only supports 1 when enabling overlap_moe_expert_parallel_comm").
    m.mtp_num_layers = None
    for f in ("account_for_embedding_in_pipeline_split", "account_for_loss_in_pipeline_split"):
        if hasattr(m, f):
            setattr(m, f, False)
    for f in ("num_layers_in_first_pipeline_stage", "num_layers_in_last_pipeline_stage"):
        if hasattr(m, f):
            setattr(m, f, None)

    # Keep the mock dataset's sequence length aligned with the model (guarded — field name varies).
    for ds_field in ("sequence_length", "seq_length"):
        if hasattr(cfg.dataset, ds_field):
            setattr(cfg.dataset, ds_field, seq_len)

    # 4) train / batch
    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = global_batch
    cfg.train.micro_batch_size = micro_batch

    # 5) the single A/B toggle (identical logic + B300-allowlist guard as bench_dsv3_pretrain.py).
    dispatcher = os.environ.get("MOE_DISPATCHER", "deepep").lower()
    if dispatcher == "alltoall":
        m.moe_token_dispatcher_type = "alltoall"          # NCCL all-to-all over EFA (baseline)
        m.moe_flex_dispatcher_backend = None
    elif dispatcher == "deepep":
        m.moe_flex_dispatcher_backend = "deepep"          # flex + deepep -> UCCL deep_ep over EFA
        apply_flex_dispatcher_backend(m, "deepep")        # sets type="flex" + clears shared-expert overlap
        # A/B VALIDITY GUARD. apply_flex_dispatcher_backend early-returns (leaving type != "flex")
        # if the device-name allowlist doesn't match, which would silently run the deepep arm as
        # plain alltoall and zero out the A/B delta. Fail loudly instead.
        if m.moe_token_dispatcher_type != "flex":
            raise RuntimeError(
                "deepep arm did not become flex (got %r): apply_flex_dispatcher_backend "
                "early-returned — device %r not in the B200/B300 allowlist. The deepep A/B arm "
                "would silently run alltoall; aborting to avoid an invalid A/B."
                % (m.moe_token_dispatcher_type, torch.cuda.get_device_properties(0).name)
            )
    else:
        raise ValueError("MOE_DISPATCHER must be 'alltoall' or 'deepep', got %r" % dispatcher)

    # moe_shared_expert_overlap is alltoall-only; hold OFF on BOTH arms to isolate the dispatcher.
    if hasattr(m, "moe_shared_expert_overlap"):
        m.moe_shared_expert_overlap = False

    # 6) Forced router load-balancing (representative regime). Random-init router routes
    #    pathologically -> ~18x stalls that are an artifact of the untrained router, not the
    #    dispatcher. Held IDENTICAL across arms. Override off with MOE_FORCE_BALANCE=off.
    if os.environ.get("MOE_FORCE_BALANCE", "on").lower() == "on":
        if hasattr(m, "moe_router_force_load_balancing"):
            m.moe_router_force_load_balancing = True

    # 7) A2A/EP overlap — held IDENTICAL across arms within a run. overlap=on enables
    #    overlap_moe_expert_parallel_comm (1F1B hides the EP all-to-all); on core 0.17.1 it needs a
    #    virtual pipeline (PP>1) and recompute fully OFF. We ALWAYS (re)derive the pipeline layout
    #    because the swapped Kimi-K2 model starts with pipeline_model_parallel_layout=None.
    overlap = os.environ.get("MOE_A2A_OVERLAP", "on").lower() == "on"
    if overlap and pp > 1:
        m.virtual_pipeline_model_parallel_size = 2        # recipe's shipped (8,2) 16-chunk layout
        m.recompute_granularity = None
        m.recompute_method = None
        m.recompute_num_layers = None
        if getattr(m, "recompute_modules", None):
            m.recompute_modules = [x for x in m.recompute_modules if x != "moe"]
    else:
        m.virtual_pipeline_model_parallel_size = None
        m.recompute_granularity = "full"                  # fit activation memory at 384 experts
        m.recompute_method = "uniform"
        m.recompute_num_layers = 1
    set_deepseek_v3_pipeline_model_parallel_layout(m)     # mtp=0 -> last stage ["loss"]

    for obj in (getattr(cfg, "comm_overlap", None), m):
        if obj is None:
            continue
        if hasattr(obj, "overlap_moe_expert_parallel_comm"):
            obj.overlap_moe_expert_parallel_comm = overlap
        if hasattr(obj, "delay_wgrad_compute"):
            obj.delay_wgrad_compute = False

    # 8) Per-iteration loss logging (the loss-equivalence curve source) + analytical throughput.
    if hasattr(cfg, "logger"):
        if hasattr(cfg.logger, "log_throughput"):
            cfg.logger.log_throughput = True
        if hasattr(cfg.logger, "log_interval"):
            cfg.logger.log_interval = 1

    logger.info(
        "bench cfg (KIMI-K2): dispatcher=%s overlap=%s | L=%s h=%s experts=%s topk=%s "
        "n_group=%s heads=%s mtp=%s MLA=%s | TP%s PP%s EP%s CP%s | iters=%s gbs=%s mbs=%s seq=%s",
        dispatcher, overlap, m.num_layers, m.hidden_size, m.num_moe_experts, m.moe_router_topk,
        getattr(m, "moe_router_num_groups", "?"), m.num_attention_heads, m.mtp_num_layers,
        getattr(m, "multi_latent_attention", "?"),
        tp, pp, ep, cp, train_iters, global_batch, micro_batch, seq_len,
    )
    return cfg


def main():
    from megatron.bridge.training.gpt_step import forward_step as _forward_step
    from megatron.bridge.training.pretrain import pretrain

    fwd = _forward_step
    # LOSS_PROBE=1: wrap the loss func to print per-microbatch loss on the last PP stage (used for
    # the fine iteration-1 work-equivalence check). Per-iteration curve comes from Megatron's own
    # `lm loss` log (log_interval=1) and does NOT need this. Identical to bench_dsv3_pretrain.py.
    if os.environ.get("LOSS_PROBE") == "1":
        _n = {"i": 0}

        def fwd(state, data_iterator, model, return_schedule_plan=False):
            out, loss_fn = _forward_step(state, data_iterator, model, return_schedule_plan)

            def wrapped(*a, **k):
                res = loss_fn(*a, **k)
                try:
                    loss_sum = float(res[0].detach().float().item())
                    ntok = float(res[1].item()) if len(res) > 1 and res[1] is not None else float("nan")
                    mean = loss_sum / ntok if ntok == ntok and ntok else float("nan")
                    _n["i"] += 1
                    print("[LOSSPROBE] call=%d loss_sum=%.6f num_tokens=%.0f mean_loss=%.6f"
                          % (_n["i"], loss_sum, ntok, mean), flush=True)
                except Exception as e:  # never let the probe break the run
                    print("[LOSSPROBE] err %r" % (e,), flush=True)
                return res

            return out, wrapped

    cfg = build_config()
    pretrain(config=cfg, forward_step_func=fwd)


if __name__ == "__main__":
    main()
