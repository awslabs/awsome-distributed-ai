# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tier-B throughput A/B/C entrypoint: Qwen3-235B-A22B MoE pretrain step.

Builds the real Qwen3-235B-A22B architecture (128-expert fine-grained MoE, top-8,
94 layers, hidden 4096, no shared expert) via Megatron-Bridge's shipped recipe with
**mock data** and **random-init weights**, then runs ``pretrain()`` for a fixed number
of iterations. The ONLY thing that changes between the benchmark arms is the MoE token
dispatcher, selected by ``MOE_DISPATCHER`` — and, for the ``deepep`` arm, *which*
``deep_ep`` the image provides:

    MOE_DISPATCHER=alltoall  -> moe_token_dispatcher_type="alltoall"   (NCCL all-to-all / EFA) [baseline]
    MOE_DISPATCHER=deepep    -> flex + moe_flex_dispatcher_backend="deepep"                      [treatment]
                                 transport = whatever `import deep_ep` resolves to in the IMAGE:
                                   * UCCL image    -> UCCL EFA-native deep_ep
                                   * NVSHMEM image -> NVIDIA DeepEP over NVSHMEM-libfabric/EFA

So the THREE-way comparison (NCCL vs DeepEP+UCCL vs DeepEP+NVSHMEM) is two images ×
the same two MOE_DISPATCHER values; this script is identical across all three arms.

Why this is a valid A/B (../../README.md): Megatron's throughput numerator is analytical
(FLOPs from config), and model/data/parallelism/precision/seed are byte-identical across
arms, so the iter-time ratio isolates the dispatcher. Random init + mock data are sound
because we measure step time, not loss — and they decouple the A/B from the HF checkpoint.

Grounded against the image (Megatron-Bridge 0.4.2 / Megatron-Core 0.17.1, nemo:26.04.01):
- recipe   : megatron.bridge.recipes.qwen.qwen3_moe.qwen3_235b_a22b_pretrain_config
             (no-arg; mock data by default; calls apply_flex_dispatcher_backend ITSELF)
- toggle   : megatron.bridge.training.flex_dispatcher_backend.apply_flex_dispatcher_backend
- fwd step : megatron.bridge.training.gpt_step.forward_step
- launch   : megatron.bridge.training.pretrain.pretrain(config=cfg, forward_step_func=...)

All knobs come from env so the manifest is the single source of truth and arms differ only
in MOE_DISPATCHER (and MOE_A2A_OVERLAP, held identical across arms within a run).
"""

import logging
import os

logger = logging.getLogger("bench_qwen3_pretrain")
logging.basicConfig(level=logging.INFO)


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def build_config():
    # The recipe builder takes NO arguments — it returns a fully-populated
    # ConfigContainer (mock data on, Qwen3-235B-A22B 94-layer/128-expert provider,
    # TP4/PP16/CP2/EP8 defaults) and calls apply_flex_dispatcher_backend(cfg.model,
    # "deepep") itself at the end. We build it, then MUTATE cfg fields for our A/B shape.
    from megatron.bridge.recipes.qwen.qwen3_moe import qwen3_235b_a22b_pretrain_config
    from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend

    # Parallelism — manifest contract. 64-GPU (8x p6-b300) EP sweep:
    #   EP16: TP8 * PP4 * DP2 (CP1) -> EP16 divides TP*DP=16 and 128 experts (8/rank).
    #   EP32: TP8 * PP2 * DP4 (CP1) -> EP32 divides TP*DP=32 and 128 experts (4/rank).
    tp = _int("TENSOR_PARALLEL", 8)
    pp = _int("PIPELINE_PARALLEL", 2)
    ep = _int("EXPERT_PARALLEL", 32)
    cp = _int("CONTEXT_PARALLEL", 1)

    train_iters = _int("TRAIN_ITERS", 24)
    global_batch = _int("GLOBAL_BATCH", 256)
    micro_batch = _int("MICRO_BATCH", 1)
    seq_len = _int("SEQ_LEN", 4096)

    cfg = qwen3_235b_a22b_pretrain_config()   # no-arg; mock data by default
    m = cfg.model

    # ---- override parallelism / iters / batch for our shape --------------------
    # Recipe defaults to TP4/PP16/CP2/EP8 (a many-node layout); we use a 64-GPU layout
    # with TP inside one NVLink node (TP8) and EP spanning nodes.
    m.tensor_model_parallel_size = tp
    m.pipeline_model_parallel_size = pp
    m.expert_model_parallel_size = ep
    m.context_parallel_size = cp
    m.expert_tensor_parallel_size = 1            # ETP=1 so EP divides TP*DP
    m.seq_length = seq_len
    # The dataset sequence length must match the model's, or cfg.validate() aborts
    # ("sequence length configuration in model config and dataset config match"). The
    # recipe defaults both to 4096; keep them in sync whenever SEQ_LEN is overridden.
    if hasattr(cfg, "dataset") and hasattr(cfg.dataset, "sequence_length"):
        cfg.dataset.sequence_length = seq_len
    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = global_batch
    cfg.train.micro_batch_size = micro_batch

    # This is a throughput benchmark — never write checkpoints. pretrain() otherwise saves a
    # full 235B checkpoint at the end of every run (slow + fills disk over a multi-run sweep).
    if hasattr(cfg, "checkpoint"):
        cfg.checkpoint.save = None
        cfg.checkpoint.load = None
        if hasattr(cfg.checkpoint, "save_interval"):
            cfg.checkpoint.save_interval = None
    # Disable the post-training evaluation loop (the recipe runs ~32 eval iters, which on a
    # 235B model dwarfs the few training iters we time and can blow the run timeout). We only
    # measure training-step time.
    if hasattr(cfg, "train"):
        if hasattr(cfg.train, "eval_iters"):
            cfg.train.eval_iters = 0
        if hasattr(cfg.train, "eval_interval"):
            cfg.train.eval_interval = train_iters + 1000

    # Expert count: keep the recipe-native 128 (Qwen3-235B-A22B). The dispatcher A/B/C
    # does not depend on the exact expert count; opt-in override only.
    if os.environ.get("NUM_MOE_EXPERTS"):
        m.num_moe_experts = _int("NUM_MOE_EXPERTS", m.num_moe_experts)

    # Optional layer-count override (default: recipe-native 94 = real Qwen3-235B).
    if os.environ.get("NUM_LAYERS"):
        m.num_layers = _int("NUM_LAYERS", m.num_layers)

    # ---- the single A/B toggle -------------------------------------------------
    # apply_flex_dispatcher_backend(model, backend) sets BOTH the type ("flex") and
    # backend, AND clears moe_shared_expert_overlap (alltoall-only). The alltoall arm
    # just sets the type back and disables the flex backend.
    dispatcher = os.environ.get("MOE_DISPATCHER", "deepep").lower()
    if dispatcher == "alltoall":
        m.moe_token_dispatcher_type = "alltoall"          # NCCL all-to-all over EFA (baseline)
        m.moe_flex_dispatcher_backend = None
    elif dispatcher == "deepep":
        m.moe_flex_dispatcher_backend = "deepep"          # flex + deepep -> deep_ep over EFA
        apply_flex_dispatcher_backend(m, "deepep")        # sets type="flex" + clears shared-expert overlap
        # A/B VALIDITY GUARD. apply_flex_dispatcher_backend EARLY-RETURNS (leaving
        # type != "flex") if the device-name allowlist (.startswith("NVIDIA B300"))
        # doesn't match. That would silently run the deepep arm as plain alltoall and
        # zero out the delta. Fail loudly instead of producing a fake null result.
        if m.moe_token_dispatcher_type != "flex":
            import torch
            raise RuntimeError(
                "deepep arm did not become flex (got %r): apply_flex_dispatcher_backend "
                "early-returned — device %r not in the B200/B300 allowlist. Aborting to "
                "avoid an invalid A/B."
                % (m.moe_token_dispatcher_type, torch.cuda.get_device_properties(0).name)
            )
    else:
        raise ValueError("MOE_DISPATCHER must be 'alltoall' or 'deepep', got %r" % dispatcher)

    # moe_shared_expert_overlap is alltoall-only AND Qwen3 has no shared expert; hold it
    # OFF on both arms (recipe default is already False).
    if hasattr(m, "moe_shared_expert_overlap"):
        m.moe_shared_expert_overlap = False

    # ---- Forced router load-balancing (representative dispatcher regime) -------
    # With random-init weights + mock data the router is DEGENERATE (tokens pile onto a
    # few experts -> one EP rank's all-to-all floods while the rest idle -> bimodal stalls
    # that are an artifact of the untrained router, not the dispatcher). Real training
    # stays balanced via the aux loss; moe_router_force_load_balancing reproduces that.
    # Held IDENTICAL across arms. (Recipe default False.) Disable with MOE_FORCE_BALANCE=off.
    if os.environ.get("MOE_FORCE_BALANCE", "on").lower() == "on":
        if hasattr(m, "moe_router_force_load_balancing"):
            m.moe_router_force_load_balancing = True

    # ---- A2A/EP overlap — held IDENTICAL across arms within a run --------------
    # MOE_A2A_OVERLAP=on enables overlap_moe_expert_parallel_comm (1F1B hides the EP
    # all-to-all behind compute — the deployment regime). On core 0.17.1 this has hard
    # co-requirements when PP>1: a virtual pipeline (VPP) and recomputation fully OFF.
    #
    # Qwen3 is a UNIFORM 94-layer stack with NO shipped VPP layout helper (unlike DSV3's
    # set_deepseek_v3_pipeline_model_parallel_layout). Standard VPP needs
    # num_layers % (pp*vpp) == 0, and 94 = 2*47 is not divisible. For overlap=on we
    # therefore round num_layers UP to the nearest pp*vpp multiple (e.g. 94 -> 96) and
    # disable the embedding/loss pipeline-split accounting so a plain uniform split
    # applies. This makes overlap=on a SEPARATE within-regime A/B (different num_layers
    # AND recompute vs overlap=off) — NEVER subtract a number across the two regimes.
    overlap = os.environ.get("MOE_A2A_OVERLAP", "on").lower() == "on"
    if overlap:
        if pp > 1:
            vpp = _int("VPP", 2)
            m.virtual_pipeline_model_parallel_size = vpp
            block = pp * vpp
            if m.num_layers % block != 0:
                new_nl = ((m.num_layers // block) + 1) * block
                logger.warning(
                    "overlap=on: rounding num_layers %d -> %d for VPP=%d/PP=%d divisibility "
                    "(Qwen3 has no shipped VPP layout helper; this is a separate within-regime A/B)",
                    m.num_layers, new_nl, vpp, pp,
                )
                m.num_layers = new_nl
            for attr in ("account_for_embedding_in_pipeline_split",
                         "account_for_loss_in_pipeline_split"):
                if hasattr(m, attr):
                    setattr(m, attr, False)
        # Recomputation must be fully disabled for the overlap path.
        m.recompute_granularity = None
        m.recompute_method = None
        m.recompute_num_layers = None
        if getattr(m, "recompute_modules", None):
            m.recompute_modules = [x for x in m.recompute_modules if x != "moe"]
    # Set the overlap flag on whichever config object exposes it. Keep delay_wgrad_compute
    # OFF to isolate the overlap mechanism.
    for obj in (getattr(cfg, "comm_overlap", None), m):
        if obj is None:
            continue
        if hasattr(obj, "overlap_moe_expert_parallel_comm"):
            obj.overlap_moe_expert_parallel_comm = overlap
        if hasattr(obj, "delay_wgrad_compute"):
            obj.delay_wgrad_compute = False

    # Optional activation recomputation, independent of the dispatcher (held identical across
    # arms). overlap=on requires recompute OFF (forced above), so only apply when overlap is
    # off. Needed to fit large per-stage layer counts on few nodes — e.g. EP32 at PP1 on 4
    # nodes (all 94 layers on one stage). RECOMPUTE=full|selective.
    recompute = os.environ.get("RECOMPUTE", "").lower()
    if not overlap and recompute in ("full", "selective"):
        m.recompute_granularity = "full" if recompute == "full" else "selective"
        if recompute == "full":
            m.recompute_method = "uniform"
            m.recompute_num_layers = _int("RECOMPUTE_NUM_LAYERS", 1)
        logger.info("recompute enabled: granularity=%s", m.recompute_granularity)

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
    # LOSS_PROBE=1: wrap the loss function to print per-microbatch loss on the last PP
    # stage. Used for the work-equivalence A/B check (deepep vs alltoall, and UCCL vs
    # NVSHMEM, must yield identical loss on identical data/seed/init — a dispatcher that
    # dropped or mis-routed tokens would diverge). This is the guard for the NVSHMEM arm's
    # IBGDA->host-proxy + put_signal patches.
    if os.environ.get("LOSS_PROBE") == "1":
        _n = {"i": 0}

        # MUST match gpt_step.forward_step's exact signature so the training loop's
        # prepare_forward_step_func() injects `state` as a partial (a variadic wrapper
        # hides that param and breaks the call arity).
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
