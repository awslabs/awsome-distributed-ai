# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tier-B throughput A/B entrypoint: DeepSeek-V3 / Kimi-K2-class MoE pretrain step.

Builds the real DeepSeek-V3 architecture (MLA + fine-grained MoE — the Kimi-K2 family)
via Megatron-Bridge's shipped recipe with **mock data** and **random-init weights**, then
runs ``pretrain()`` for a fixed number of iterations. The ONLY thing that changes between
the two benchmark arms is the MoE token dispatcher, selected by ``MOE_DISPATCHER``:

    MOE_DISPATCHER=alltoall  -> moe_token_dispatcher_type="alltoall"  (NCCL all-to-all / EFA)  [baseline]
    MOE_DISPATCHER=deepep    -> flex + moe_flex_dispatcher_backend="deepep" (UCCL EFA drop-in) [treatment]

Why this is a valid A/B (benchmarks/README.md): Megatron's throughput numerator is
analytical (FLOPs from config), and model/data/parallelism/precision/seed are byte-identical
across arms, so the iter-time ratio isolates the dispatcher. Random init + mock data are
sound because we measure step time, not loss — and they decouple the A/B from the ~2 TB
checkpoint + HF/AutoBridge path (unavailable / broken on this image).

Grounded against the image (Megatron-Bridge 0.2.0, verified by introspection):
- recipe   : megatron.bridge.recipes.deepseek.deepseek_v3.deepseek_v3_pretrain_config_32nodes
             (**DeepSeekV3CommonKwargs — MCore-style names: tensor_model_parallel_size, ...)
- toggle   : megatron.bridge.recipes.deepseek.deepseek_v3.apply_flex_dispatcher_backend
- fwd step : megatron.bridge.training.gpt_step.forward_step
- launch   : megatron.bridge.training.pretrain.pretrain(config=cfg, forward_step_func=forward_step)
  (mirrors examples/recipes/llama/pretrain_llama3_8b.py in the image)

All knobs come from env so the manifest is the single source of truth and both arms differ
only in MOE_DISPATCHER (and MOE_A2A_OVERLAP, held identical across arms within a run).
"""

import logging
import os

logger = logging.getLogger("bench_dsv3_pretrain")
logging.basicConfig(level=logging.INFO)


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def build_config():
    # API note (Megatron-Bridge 0.4.2, shipped in nemo:26.04.01): the recipe builder
    # takes NO arguments — it returns a fully-populated ConfigContainer (mock data on,
    # DSV3 61-layer pipeline layout via set_deepseek_v3_pipeline_model_parallel_layout,
    # and it calls apply_flex_dispatcher_backend(cfg.model, ...) ITSELF at the end). We
    # build it, then MUTATE cfg fields for our A/B shape. (On 0.2.0 this builder took
    # **DeepSeekV3CommonKwargs; that signature was dropped in 0.4.x.)
    from megatron.bridge.recipes.deepseek.deepseek_v3 import (
        deepseek_v3_pretrain_config_32nodes,
        apply_flex_dispatcher_backend,
    )

    # Parallelism — manifest contract. Canonical 256-GPU layout:
    # TP8 * PP8 = 64; world 256 -> DP=4; EP=32 divides TP*DP=32 (ETP=1).
    tp = _int("TENSOR_PARALLEL", 8)
    pp = _int("PIPELINE_PARALLEL", 8)
    ep = _int("EXPERT_PARALLEL", 32)
    cp = _int("CONTEXT_PARALLEL", 1)

    train_iters = _int("TRAIN_ITERS", 70)
    global_batch = _int("GLOBAL_BATCH", 512)
    micro_batch = _int("MICRO_BATCH", 1)
    seq_len = _int("SEQ_LEN", 4096)

    cfg = deepseek_v3_pretrain_config_32nodes()   # no-arg on 0.4.2; mock data by default
    m = cfg.model

    # ---- override parallelism / iters / batch for our shape --------------------
    # The recipe defaults to TP2/PP8/EP32; we use TP8/PP8/EP32 (TP inside one NVLink
    # node). seq_length and the DSV3 61-layer pipeline layout are left as the recipe set
    # them (re-running set_deepseek_v3_pipeline_model_parallel_layout is not needed since
    # PP stays 8).
    m.tensor_model_parallel_size = tp
    m.pipeline_model_parallel_size = pp
    m.expert_model_parallel_size = ep
    m.context_parallel_size = cp
    m.seq_length = seq_len
    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = global_batch
    cfg.train.micro_batch_size = micro_batch

    # Expert count: keep the recipe-native 256 (DeepSeek-V3). Overriding to Kimi-K2's
    # 384 without recomputing the node-group routing (moe_router_num_groups /
    # group_topk) breaks the build. The dispatcher A/B does not depend on the exact
    # expert count, so 256 is a valid Kimi-K2-FAMILY substrate. Opt-in override only.
    if os.environ.get("NUM_MOE_EXPERTS"):
        m.num_moe_experts = _int("NUM_MOE_EXPERTS", m.num_moe_experts)

    # ---- the single A/B toggle -------------------------------------------------
    # On 0.4.2 apply_flex_dispatcher_backend(model, backend) sets BOTH the type ("flex")
    # and backend, AND clears moe_shared_expert_overlap (which is alltoall-only) — so we
    # no longer hand-patch those. The alltoall arm just sets the type and leaves overlap
    # as the recipe set it (recipe default for the baseline path).
    dispatcher = os.environ.get("MOE_DISPATCHER", "deepep").lower()
    if dispatcher == "alltoall":
        m.moe_token_dispatcher_type = "alltoall"          # NCCL all-to-all over EFA (baseline)
        m.moe_flex_dispatcher_backend = None
    elif dispatcher == "deepep":
        m.moe_flex_dispatcher_backend = "deepep"          # flex + deepep -> UCCL deep_ep over EFA
        apply_flex_dispatcher_backend(m, "deepep")        # sets type="flex" + clears shared-expert overlap
        # A/B VALIDITY GUARD. apply_flex_dispatcher_backend EARLY-RETURNS (leaving
        # type != "flex") if the device-name allowlist (.startswith("NVIDIA B300"))
        # doesn't match — e.g. an unexpected device-name string or MIG. That would
        # silently run the deepep arm as plain alltoall and zero out the A/B delta,
        # and validate_flex_dispatcher_backend() would NOT catch it (it only fires
        # when type IS "flex"). Fail loudly instead of producing a fake null result.
        if m.moe_token_dispatcher_type != "flex":
            import torch
            raise RuntimeError(
                "deepep arm did not become flex (got %r): apply_flex_dispatcher_backend "
                "early-returned — device %r not in the B200/B300 allowlist. The deepep A/B "
                "arm would silently run alltoall; aborting to avoid an invalid A/B."
                % (m.moe_token_dispatcher_type, torch.cuda.get_device_properties(0).name)
            )
    else:
        raise ValueError("MOE_DISPATCHER must be 'alltoall' or 'deepep', got %r" % dispatcher)

    # moe_shared_expert_overlap is alltoall-only. Hold it OFF on BOTH arms so the A/B
    # isolates the dispatcher (the deepep path can't use it; giving it to alltoall only
    # would bias the baseline). apply_flex_dispatcher_backend already cleared it for the
    # deepep arm; clear it explicitly for the alltoall arm too.
    if hasattr(m, "moe_shared_expert_overlap"):
        m.moe_shared_expert_overlap = False

    # ---- Forced router load-balancing (representative dispatcher regime) -------
    # With random-init weights + mock data the learned router is DEGENERATE: it
    # routes pathologically (tokens pile onto a few experts), so one EP rank's
    # all-to-all floods while the rest idle -> bimodal ~18x step-time stalls that
    # are an artifact of the untrained router, not the dispatcher. Real training
    # stays ~balanced via the aux load-balancing loss; moe_router_force_load_balancing
    # reproduces that balanced regime so the A/B measures dispatcher throughput on a
    # representative token distribution. Held IDENTICAL across arms. (Recipe default
    # is False.) Override off with MOE_FORCE_BALANCE=off only to study the imbalance.
    if os.environ.get("MOE_FORCE_BALANCE", "on").lower() == "on":
        if hasattr(m, "moe_router_force_load_balancing"):
            m.moe_router_force_load_balancing = True

    # ---- A2A/EP overlap — held IDENTICAL across arms within a run --------------
    # MOE_A2A_OVERLAP=on enables overlap_moe_expert_parallel_comm (1F1B hides the EP
    # all-to-all behind compute — the deployment-realistic regime). On core 0.17.1 this
    # has hard co-requirements (transformer_config.py validate): with PP>1 it needs a
    # virtual pipeline (VPP), and recomputation must be fully OFF. We reconfigure BOTH
    # arms identically, so overlap=on is a SEPARATE, internally-valid A/B — NOT directly
    # comparable to overlap=off (which keeps VPP=1 + full recompute). Never subtract a
    # number across the two regimes.
    overlap = os.environ.get("MOE_A2A_OVERLAP", "on").lower() == "on"
    if overlap:
        if pp > 1:
            # VPP required when PP>1. Use the recipe's own (pp_size, vp_size) layout map:
            # (8,2) is a shipped 16-chunk layout for the 61-layer DSV3 stack.
            from megatron.bridge.recipes.deepseek.deepseek_v3 import (
                set_deepseek_v3_pipeline_model_parallel_layout,
            )
            m.virtual_pipeline_model_parallel_size = 2
            set_deepseek_v3_pipeline_model_parallel_layout(m)  # re-derive layout for VPP=2
        # Recomputation must be fully disabled for the overlap path.
        m.recompute_granularity = None
        m.recompute_method = None
        m.recompute_num_layers = None
        if getattr(m, "recompute_modules", None):
            m.recompute_modules = [x for x in m.recompute_modules if x != "moe"]
    # Set the overlap flag on whichever config object exposes it (model config is what the
    # validator reads; comm_overlap mirrors it). Keep delay_wgrad_compute OFF to isolate the
    # overlap mechanism and minimise the constraint surface.
    for obj in (getattr(cfg, "comm_overlap", None), m):
        if obj is None:
            continue
        if hasattr(obj, "overlap_moe_expert_parallel_comm"):
            obj.overlap_moe_expert_parallel_comm = overlap
        if hasattr(obj, "delay_wgrad_compute"):
            obj.delay_wgrad_compute = False

    # Ensure the analytical throughput line is emitted (RESULTS scraping keys on it).
    if hasattr(cfg, "logger"):
        if hasattr(cfg.logger, "log_throughput"):
            cfg.logger.log_throughput = True
        if hasattr(cfg.logger, "log_interval"):
            cfg.logger.log_interval = 1

    logger.info(
        "bench cfg: dispatcher=%s overlap=%s | L=%s h=%s experts=%s topk=%s | "
        "TP%s PP%s EP%s CP%s | iters=%s gbs=%s mbs=%s seq=%s",
        dispatcher, overlap, m.num_layers, m.hidden_size, m.num_moe_experts,
        m.moe_router_topk, tp, pp, ep, cp, train_iters, global_batch, micro_batch, seq_len,
    )
    return cfg


def main():
    from megatron.bridge.training.gpt_step import forward_step as _forward_step
    from megatron.bridge.training.pretrain import pretrain

    fwd = _forward_step
    # LOSS_PROBE=1: wrap the loss function to print per-microbatch loss on whatever rank
    # computes it (the last pipeline stage). Used only for the work-equivalence A/B check
    # (deepep vs alltoall must yield identical loss on identical data/seed/init — a
    # dispatcher that dropped tokens or mis-routed would diverge). gpt_step's loss func
    # (masked_next_token_loss) returns (loss_sum, num_tokens, {"lm loss": ...}); mean
    # per-token loss = loss_sum / num_tokens is the arm-comparable quantity.
    if os.environ.get("LOSS_PROBE") == "1":
        _n = {"i": 0}

        # MUST match gpt_step.forward_step's exact signature
        # (state, data_iterator, model, return_schedule_plan=False): the training loop's
        # prepare_forward_step_func() inspects the signature and injects `state` as a
        # partial only when it sees a `state` parameter. A variadic (*args) wrapper hides
        # that param, so state is not injected and the call arity is wrong.
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
