# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Throughput + loss-equivalence A/B entrypoint: Kimi-K2 384-expert MoE pretrain step.

Builds the LITERAL Kimi-K2 architecture (384 routed experts, 64 attention heads, node-group
routing n_group=1, MLA, 61 layers, NO MTP) via Megatron-Bridge's AutoBridge from the HF config,
with **mock data** and **random-init weights**, then runs ``pretrain()`` for a fixed number of
iterations. ``EP_ARM`` is the only dispatcher selector and is checked against the immutable
``/opt/benchmark/backend.json`` identity embedded in each image.

This mirrors ``../../dsv3/benchmarks/bench_dsv3_pretrain.py`` but swaps the recipe-native DeepSeek-V3
256-expert model for the real Kimi-K2 provider:

- the DSV3 recipe ``deepseek_v3_pretrain_config_32nodes()`` supplies the model-agnostic
  scaffolding (MOCK dataset, training/optimizer/ddp/logger config, seed);
- ``AutoBridge.from_hf_pretrained(<hf>, trust_remote_code=True).to_megatron_provider(load_weights=False)``
  supplies the correct Kimi-K2 model provider (all coupled MLA / expert / group dims read from
  the HF config); we graft it onto ``cfg.model`` and re-derive the 61-layer pipeline layout.

The locked NeMo 26.08 image routes Kimi-K2-Base
(``architectures=["DeepseekV3ForCausalLM"]``) to DeepSeekV3Bridge and yields
num_moe_experts=384, moe_router_num_groups=1, num_attention_heads=64, num_layers=61,
multi_latent_attention=True, mtp_num_layers=0. ``trust_remote_code=True`` is required because the
HF config uses auto_map -> configuration_deepseek.DeepseekV3Config custom code.

Model, data, parallelism, precision, and seed are held identical across all 4 arms. The
per-iteration ``lm loss`` line (log_interval=1) is retained as a validity signal. The launcher is
the single source of truth, and backend overlays alter only EP_ARM, image, backend environment,
and required device mounts.
"""

import hashlib
import json
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.bridge.training.callbacks import Callback, CallbackContext

logger = logging.getLogger("bench_kimi_k2_pretrain")
logging.basicConfig(level=logging.INFO)

# HF Kimi-K2 repo dir staged on FSx (config.json + configuration_deepseek.py; weights unused).
HF_PATH = os.environ.get("KIMI_K2_HF_PATH", "moonshotai/Kimi-K2-Base")
HF_REVISION = os.environ.get(
    "KIMI_K2_REVISION", "ce72df012259dcc55d945e890f815fe7ef69159c"
)


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


class RuntimeDispatcherIdentity(Callback):
    """Fail closed unless the built model owns the requested dispatcher objects."""

    def __init__(self) -> None:
        self.elastic_buffer_observed = False

    @staticmethod
    def _model_chunks(context: CallbackContext):
        if isinstance(context.model, (list, tuple)):
            return context.model
        return [context.model]

    def on_train_start(self, context: CallbackContext) -> None:
        from megatron.core.transformer.moe.moe_layer import BaseMoELayer
        from megatron.core.transformer.moe.token_dispatcher import (
            MoEAlltoAllTokenDispatcher,
            MoEFlexTokenDispatcher,
            _DeepepManager,
            _DeepepV2Manager,
        )

        arm = os.environ["EP_ARM"]
        moe_layers = [
            module
            for chunk in self._model_chunks(context)
            for module in chunk.modules()
            if isinstance(module, BaseMoELayer)
        ]
        if not moe_layers:
            raise RuntimeError(f"{arm} built no BaseMoELayer instances")
        dispatchers = [
            layer.token_dispatcher
            for layer in moe_layers
            if layer.token_dispatcher is not None
        ]
        if len(dispatchers) != len(moe_layers):
            raise RuntimeError(
                f"{arm} left MoE layers without token dispatchers: "
                f"layers={len(moe_layers)} dispatchers={len(dispatchers)}"
            )

        if arm == "nccl-alltoall":
            wrong_dispatchers = [
                type(dispatcher).__name__
                for dispatcher in dispatchers
                if not isinstance(dispatcher, MoEAlltoAllTokenDispatcher)
            ]
            if wrong_dispatchers:
                raise RuntimeError(
                    "legacy all-to-all arm selected unexpected dispatchers: "
                    f"expected={MoEAlltoAllTokenDispatcher.__name__} actual={wrong_dispatchers}"
                )
            expected_manager = None
        elif arm in ("uccl", "deepep-v1-nvshmem"):
            expected_manager = _DeepepManager
        elif arm == "deepep-v2-gin-gda":
            expected_manager = _DeepepV2Manager
        else:
            raise RuntimeError(f"unknown EP_ARM during runtime inspection: {arm}")

        flex_dispatchers = [
            dispatcher
            for dispatcher in dispatchers
            if isinstance(dispatcher, MoEFlexTokenDispatcher)
        ]
        managers = [dispatcher._comm_manager for dispatcher in flex_dispatchers]
        if expected_manager is not None:
            if len(flex_dispatchers) != len(dispatchers):
                actual = [type(dispatcher).__name__ for dispatcher in dispatchers]
                raise RuntimeError(
                    f"{arm} selected non-flex dispatchers: "
                    f"expected={MoEFlexTokenDispatcher.__name__} actual={actual}"
                )
            wrong = [
                type(manager).__name__
                for manager in managers
                if not isinstance(manager, expected_manager)
            ]
            if wrong:
                raise RuntimeError(
                    f"{arm} selected unexpected flex managers: expected={expected_manager.__name__} actual={wrong}"
                )

        group_sizes = sorted(
            {dist.get_world_size(layer.ep_group) for layer in moe_layers}
        )
        expected_group_size = _int("EXPERT_PARALLEL", 32)
        if group_sizes != [expected_group_size]:
            raise RuntimeError(
                f"expert dispatcher group drift: expected={[expected_group_size]} actual={group_sizes}"
            )
        payload = {
            "arm": arm,
            "dispatcher_instances_on_rank": len(dispatchers),
            "moe_layer_instances_on_rank": len(moe_layers),
            "group_sizes": group_sizes,
            "manager": expected_manager.__name__
            if expected_manager is not None
            else None,
        }
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                "RUNTIME_DISPATCHER_IDENTITY " + json.dumps(payload, sort_keys=True),
                flush=True,
            )
            if expected_manager is not None:
                backend = "deepep_v2" if arm == "deepep-v2-gin-gda" else "deepep"
                print(
                    f"EP_BACKEND_IDENTITY backend={backend} "
                    f"manager={expected_manager.__name__} group_size={group_sizes[0]}",
                    flush=True,
                )

    def on_train_step_end(self, context: CallbackContext) -> None:
        del context
        if os.environ["EP_ARM"] != "deepep-v2-gin-gda" or self.elastic_buffer_observed:
            return
        import deep_ep
        from megatron.core.transformer.moe import fused_a2a

        buffers = list(fused_a2a._elastic_buffers.values())
        if not buffers:
            raise RuntimeError(
                "DeepEP v2 completed a step without an ElasticBuffer instance"
            )
        if not all(isinstance(buffer, deep_ep.ElasticBuffer) for buffer in buffers):
            raise RuntimeError(
                "DeepEP v2 cache contains non-ElasticBuffer objects: "
                f"{[type(value).__name__ for value in buffers]}"
            )
        self.elastic_buffer_observed = True
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"ELASTIC_BUFFER_RUNTIME_IDENTITY buffer=ElasticBuffer instances={len(buffers)}",
                flush=True,
            )

    def on_train_end(self, context: CallbackContext) -> None:
        del context
        if (
            os.environ["EP_ARM"] == "deepep-v2-gin-gda"
            and not self.elastic_buffer_observed
        ):
            raise RuntimeError(
                "DeepEP v2 training ended without runtime ElasticBuffer proof"
            )


class RouteTrace(Callback):
    """Trace only discarded warmup routing while preserving a durable route hash."""

    def __init__(self, output_dir: str, max_steps: int) -> None:
        if max_steps <= 0:
            raise ValueError("route trace max_steps must be positive")
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.completed_steps = 0

    def on_train_start(self, context: CallbackContext) -> None:
        from megatron.core.transformer.moe.router_trace import (
            get_moe_router_tracer,
            init_moe_router_tracer,
        )

        if get_moe_router_tracer() is not None:
            raise RuntimeError("an MoE router tracer is already active")
        rank = dist.get_rank() if dist.is_initialized() else 0
        init_moe_router_tracer(
            output_dir=self.output_dir,
            max_steps=self.max_steps,
            rank=rank,
            training_mode=True,
        )
        tracer = get_moe_router_tracer()
        if tracer is None:
            raise RuntimeError("MCore did not initialize the MoE router tracer")
        tracer.register_hooks(context.model)
        hook_count = len(tracer._hook_handles)
        if hook_count == 0:
            raise RuntimeError("MoE router tracer found no TopKRouter modules")
        if rank == 0:
            print(
                "ROUTER_TRACE_ACTIVE "
                + json.dumps(
                    {
                        "max_steps": self.max_steps,
                        "output_dir": self.output_dir,
                        "router_hooks": hook_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    def on_train_step_end(self, context: CallbackContext) -> None:
        if self.completed_steps >= self.max_steps:
            return
        from megatron.core.transformer.moe.router_trace import get_moe_router_tracer

        tracer = get_moe_router_tracer()
        if tracer is None:
            raise RuntimeError("MoE router tracer disappeared during training")
        tracer.advance_step(int(context.state.train_state.step))
        self.completed_steps += 1

    def on_train_end(self, context: CallbackContext) -> None:
        del context
        from megatron.core.transformer.moe.router_trace import get_moe_router_tracer

        tracer = get_moe_router_tracer()
        if tracer is None:
            raise RuntimeError("MoE router tracer is missing at train end")
        tracer.flush()
        trace_path = Path(tracer.output_path)
        records = sum(1 for line in trace_path.read_text().splitlines() if line.strip())
        if records == 0:
            raise RuntimeError(f"MoE router trace is empty: {trace_path}")
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(
                "ROUTER_TRACE_COMPLETE "
                + json.dumps(
                    {
                        "completed_steps": self.completed_steps,
                        "records": records,
                        "trace_path": str(trace_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


def build_config():
    from megatron.bridge import AutoBridge
    from megatron.bridge.recipes.deepseek.deepseek_v3 import (
        deepseek_v3_pretrain_config_32nodes,
        set_deepseek_v3_pipeline_model_parallel_layout,
    )
    from megatron.bridge.training.flex_dispatcher_backend import (
        apply_flex_dispatcher_backend,
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
    distributed_timeout_minutes = _int("DISTRIBUTED_TIMEOUT_MINUTES", 30)
    if distributed_timeout_minutes <= 0:
        raise ValueError("DISTRIBUTED_TIMEOUT_MINUTES must be positive")
    # A new micro-batch shape can spend more than the upstream 10-minute default in the
    # first TorchInductor compile while adjacent pipeline stages wait in P2P operations.
    # This timeout is common to every arm and does not alter the scored steady-state steps.
    cfg.dist.distributed_timeout_minutes = distributed_timeout_minutes

    # 2) Build the literal Kimi-K2 provider from the HF config (random init; no ~2 TB weights).
    #    trust_remote_code=True is REQUIRED (config auto_map -> configuration_deepseek.DeepseekV3Config).
    k2 = AutoBridge.from_hf_pretrained(
        HF_PATH, revision=HF_REVISION, trust_remote_code=True
    ).to_megatron_provider(load_weights=False)
    cfg.model = k2
    m = cfg.model
    literal = {
        "num_layers": 61,
        "hidden_size": 7168,
        "num_moe_experts": 384,
        "moe_router_topk": 8,
    }
    drift = {
        name: (expected, getattr(m, name, None))
        for name, expected in literal.items()
        if getattr(m, name, None) != expected
    }
    if drift:
        raise RuntimeError(
            f"Kimi-K2 architecture drift at revision {HF_REVISION}: {drift}"
        )

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
        m.cuda_graph_impl = "none"  # CUDA graphs + EP all-to-all do not mix well
    m.moe_grouped_gemm = True
    m.moe_permute_fusion = True
    # Kimi-K2 has NO multi-token-prediction layer (DSV3 ships MTP=1). Use None — NOT 0:
    # the layout helper treats None as no-MTP (`None or 0` -> ["loss"] tail), and core's
    # comm-overlap setup asserts `mtp_num_layers is None or == 1` when
    # overlap_moe_expert_parallel_comm is enabled — an int 0 trips that assert
    # ("MTP layernum only supports 1 when enabling overlap_moe_expert_parallel_comm").
    m.mtp_num_layers = None
    for f in (
        "account_for_embedding_in_pipeline_split",
        "account_for_loss_in_pipeline_split",
    ):
        if hasattr(m, f):
            setattr(m, f, False)
    for f in (
        "num_layers_in_first_pipeline_stage",
        "num_layers_in_last_pipeline_stage",
    ):
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

    # Benchmark runs consume random-init mock data and score training-step time. Do not write a
    # final ~1T-parameter checkpoint or run the recipe's post-training validation/test passes:
    # both happen outside the measured iterations, hold all 256 GPUs, and add minutes of work to
    # every arm without contributing a correctness or performance signal.
    if hasattr(cfg, "checkpoint"):
        cfg.checkpoint.save = None
        cfg.checkpoint.load = None
        if hasattr(cfg.checkpoint, "save_interval"):
            cfg.checkpoint.save_interval = None
    if hasattr(cfg, "validation"):
        cfg.validation.eval_iters = 0
        cfg.validation.eval_interval = train_iters + 1000

    performance_seed = _int("PERFORMANCE_SEED", 1234)
    cfg.rng.seed = performance_seed
    if hasattr(cfg.dataset, "random_seed"):
        cfg.dataset.random_seed = performance_seed
    # 5) One explicit four-arm selector. Image identity is independent evidence that a
    # deepep-compatible import did not silently select the wrong implementation.
    arm = os.environ["EP_ARM"]
    identity_path = os.environ.get("EP_BACKEND_IDENTITY", "/opt/benchmark/backend.json")
    with open(identity_path, encoding="utf-8") as stream:
        identity_bytes = stream.read().encode()
    identity = json.loads(identity_bytes)
    if identity.get("ep_arm") != arm:
        raise RuntimeError(
            f"EP_ARM/image mismatch: requested={arm!r}, identity={identity!r}"
        )

    if arm == "nccl-alltoall":
        m.moe_token_dispatcher_type = "alltoall"
        m.moe_flex_dispatcher_backend = None
    elif arm in ("uccl", "deepep-v1-nvshmem"):
        apply_flex_dispatcher_backend(m, "deepep")
        assert m.moe_token_dispatcher_type == "flex"
        assert m.moe_flex_dispatcher_backend == "deepep"
    elif arm == "deepep-v2-gin-gda":
        apply_flex_dispatcher_backend(m, "deepep_v2")
        assert m.moe_token_dispatcher_type == "flex"
        assert m.moe_flex_dispatcher_backend == "deepep_v2"
        m.moe_deepep_v2_num_qps = _int("DEEPEP_V2_NUM_QPS", 0)
        m.moe_deepep_v2_deterministic = False
        m.moe_deepep_v2_allow_multiple_reduction = True
        m.moe_deepep_v2_prefer_overlap_with_compute = True
    else:
        raise ValueError(f"Unknown EP_ARM: {arm}")

    logger.info(
        "EP_BACKEND_REQUEST arm=%s dispatcher=%s backend=%s image_identity_sha256=%s",
        arm,
        m.moe_token_dispatcher_type,
        m.moe_flex_dispatcher_backend,
        hashlib.sha256(identity_bytes).hexdigest(),
    )
    logger.info(
        "NO_TOKEN_DROP_CONFIG capacity_factor=%r token_dropping=%r experts=%d topk=%d",
        getattr(m, "moe_expert_capacity_factor", None),
        getattr(m, "moe_token_dropping", False),
        m.num_moe_experts,
        m.moe_router_topk,
    )
    logger.info(
        "EXPERT_GROUP_EXPECTATION expert_tensor_parallel=%d expert_parallel=%d group_size=%d",
        m.expert_tensor_parallel_size,
        m.expert_model_parallel_size,
        m.expert_tensor_parallel_size * m.expert_model_parallel_size,
    )
    print(
        "EP_BACKEND_REQUEST arm=%s dispatcher=%s backend=%s image_identity_sha256=%s"
        % (
            arm,
            m.moe_token_dispatcher_type,
            m.moe_flex_dispatcher_backend,
            hashlib.sha256(identity_bytes).hexdigest(),
        ),
        flush=True,
    )
    print(
        "NO_TOKEN_DROP_CONFIG capacity_factor=%r token_dropping=%r experts=%d topk=%d"
        % (
            getattr(m, "moe_expert_capacity_factor", None),
            getattr(m, "moe_token_dropping", False),
            m.num_moe_experts,
            m.moe_router_topk,
        ),
        flush=True,
    )
    print(
        "EXPERT_GROUP_EXPECTATION expert_tensor_parallel=%d expert_parallel=%d group_size=%d"
        % (
            m.expert_tensor_parallel_size,
            m.expert_model_parallel_size,
            m.expert_tensor_parallel_size * m.expert_model_parallel_size,
        ),
        flush=True,
    )

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
        m.virtual_pipeline_model_parallel_size = (
            2  # recipe's shipped (8,2) 16-chunk layout
        )
        m.recompute_granularity = None
        m.recompute_method = None
        m.recompute_num_layers = None
        if getattr(m, "recompute_modules", None):
            m.recompute_modules = [x for x in m.recompute_modules if x != "moe"]
    else:
        m.virtual_pipeline_model_parallel_size = None
        m.recompute_granularity = "full"  # fit activation memory at 384 experts
        m.recompute_method = "uniform"
        m.recompute_num_layers = 1
    set_deepseek_v3_pipeline_model_parallel_layout(m)  # mtp=0 -> last stage ["loss"]

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
        "bench cfg (KIMI-K2): arm=%s overlap=%s | L=%s h=%s experts=%s topk=%s "
        "n_group=%s heads=%s mtp=%s MLA=%s | TP%s PP%s EP%s CP%s | iters=%s gbs=%s mbs=%s seq=%s "
        "distributed_timeout_minutes=%s",
        arm,
        overlap,
        m.num_layers,
        m.hidden_size,
        m.num_moe_experts,
        m.moe_router_topk,
        getattr(m, "moe_router_num_groups", "?"),
        m.num_attention_heads,
        m.mtp_num_layers,
        getattr(m, "multi_latent_attention", "?"),
        tp,
        pp,
        ep,
        cp,
        train_iters,
        global_batch,
        micro_batch,
        seq_len,
        distributed_timeout_minutes,
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
            out, loss_fn = _forward_step(
                state, data_iterator, model, return_schedule_plan
            )

            def wrapped(*a, **k):
                res = loss_fn(*a, **k)
                try:
                    loss_sum = float(res[0].detach().float().item())
                    ntok = (
                        float(res[1].item())
                        if len(res) > 1 and res[1] is not None
                        else float("nan")
                    )
                    mean = loss_sum / ntok if ntok == ntok and ntok else float("nan")
                    _n["i"] += 1
                    print(
                        "[LOSSPROBE] call=%d loss_sum=%.6f num_tokens=%.0f mean_loss=%.6f"
                        % (_n["i"], loss_sum, ntok, mean),
                        flush=True,
                    )
                except Exception as e:  # never let the probe break the run
                    print("[LOSSPROBE] err %r" % (e,), flush=True)
                return res

            return out, wrapped

    cfg = build_config()
    callbacks: list[Callback] = [RuntimeDispatcherIdentity()]
    route_trace_dir = os.environ.get("ROUTER_TRACE_DIR")
    if route_trace_dir:
        callbacks.append(
            RouteTrace(
                route_trace_dir,
                _int("ROUTER_TRACE_MAX_TRAINING_ITERS", 1),
            )
        )
    pretrain(config=cfg, forward_step_func=fwd, callbacks=callbacks)


if __name__ == "__main__":
    main()
