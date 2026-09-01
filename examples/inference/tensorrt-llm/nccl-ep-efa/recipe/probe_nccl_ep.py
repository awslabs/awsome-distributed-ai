#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
"""probe_nccl_ep.py — does a real serve's selection path hand back an NcclEP, and
does that instance move bytes cross-node on EFA?

A serve never constructs NcclEP itself — it calls
CommunicationFactory.create_strategy(model_config, ...), which reads
TRTLLM_FORCE_COMM_METHOD, runs TRT-LLM's own _get_nccl_ep_unavailable_reason
feasibility gate, and derives every NcclEP ctor arg from model_config. So this
probe drives THAT entrypoint, with a ModelConfig stand-in carrying the same
fields a serve would (mapping with enable_attention_dp + moe_tp_size=1,
pretrained_config.hidden_size, torch_dtype, max_num_tokens, ...), then runs
dispatch/combine through whatever the factory returned and checks the combined
output against a CPU-derived oracle.

Runs under torchrun (see run-kernel-test.sh) so the strategy the factory selects
is exercised ACROSS the node boundary, not just instantiated — the defect class
this exists to catch is a first-dispatch illegal memory access that
instantiation alone never reaches.
"""
import os
import sys
import traceback

import torch
import torch.distributed as dist

# RANK/WORLD/LOCAL are the one required (torchrun-set) trio; they are read inside
# main() — under the bottom try/except — so a run outside torchrun surfaces as PROBE-EXC
# rather than a bare KeyError traceback. -1 sentinels keep log()/the except usable if the
# read never happened. The EP_* knobs below use .get() defaults, so they cannot KeyError.
RANK = WORLD = LOCAL = -1

E = int(os.environ.get("EP_EXPERTS", "128"))
TOK = int(os.environ.get("EP_TOKENS", "128"))
H = int(os.environ.get("EP_HIDDEN", "2048"))
K = int(os.environ.get("EP_TOPK", "8"))
NUM_QP = int(os.environ.get("EP_NUM_QP", "32"))


def log(msg):
    print(f"[rank{RANK}] {msg}", flush=True)


def install_mpi_shim():
    """mpi4py-free stand-in for mpi_comm(), used only for NCCL unique-id bootstrap.

    trtllm-serve runs under mpirun and gets a real MPI communicator; under
    torchrun there is none, so the handful of collective ops TRT-LLM's NcclEP
    bootstrap needs (bcast/allgather/barrier of the NCCL unique id) are backed
    by torch.distributed instead. This shims the transport of the bootstrap
    only — the EP data path under test is untouched.
    """
    import tensorrt_llm._utils as tu

    class _Shim:
        def Get_rank(self):
            return dist.get_rank()

        def Get_size(self):
            return dist.get_world_size()

        def Split(self, color, key):
            # Returns the world communicator, valid ONLY while ep_size == world_size (this
            # probe's shape — see build_model_config's moe_ep_size=WORLD): the EP group then
            # IS the world group. At any shape where ep_size < world_size this would silently
            # build the wrong group; the probe would need a real color-keyed split first.
            return self

        def bcast(self, obj, root=0):
            box = [obj if dist.get_rank() == root else None]
            dist.broadcast_object_list(box, src=root)
            return box[0]

        def allgather(self, obj):
            out = [None] * dist.get_world_size()
            dist.all_gather_object(out, obj)
            return out

        def barrier(self):
            dist.barrier()

        def Free(self):
            pass

    shim = _Shim()
    tu.mpi_comm = lambda: shim
    tu.mpi_rank = lambda: dist.get_rank()
    tu.mpi_world_size = lambda: dist.get_world_size()
    tu.mpi_barrier = lambda: dist.barrier()
    import tensorrt_llm

    for mod in (tensorrt_llm, tu):
        for name in ("mpi_comm", "mpi_rank", "mpi_world_size"):
            if hasattr(mod, name):
                setattr(mod, name, getattr(tu, name))


def build_model_config(mapping):
    """A ModelConfig with exactly the attributes the factory reads.

    Using a stand-in rather than a real ModelConfig keeps the probe independent
    of a checkpoint download, while the attribute set is taken from the factory
    source (mapping, pretrained_config.hidden_size, torch_dtype, quant_config,
    max_num_tokens, moe_max_num_tokens, use_cuda_graph,
    use_low_precision_moe_combine, moe_load_balancer).
    """

    class _Pretrained:
        hidden_size = H

    class _MC:
        pass

    mc = _MC()
    mc.mapping = mapping
    mc.pretrained_config = _Pretrained()
    mc.torch_dtype = torch.bfloat16
    mc.quant_config = None
    mc.max_num_tokens = TOK
    mc.moe_max_num_tokens = None
    mc.use_cuda_graph = False
    mc.use_low_precision_moe_combine = False
    mc.moe_load_balancer = None
    return mc


def main():
    global RANK, WORLD, LOCAL
    # Read the torchrun-set trio here (not at module scope) so a run outside torchrun
    # surfaces as PROBE-EXC via the bottom handler, not a bare KeyError at import.
    RANK = int(os.environ["RANK"])
    WORLD = int(os.environ["WORLD_SIZE"])
    LOCAL = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(LOCAL)
    dist.init_process_group("nccl", rank=RANK, world_size=WORLD)
    dev = torch.device("cuda", LOCAL)
    os.environ.setdefault("NCCL_EP_NUM_QP_PER_RANK", str(NUM_QP))
    log(f"probe start world={WORLD} algo={os.environ.get('TRTLLM_NCCL_EP_ALGO', '(upstream default)')}")

    # EP requires the experts to shard evenly across ranks. At an indivisible world size
    # (e.g. a 3-node run, WORLD=24, E=128) the E//WORLD floor orphans the tail experts:
    # tokens still route to them, their contributions never enter moe_out, and the probe
    # would report a false PROBE-MISMATCH on a healthy fabric. E and WORLD are rank-invariant
    # (env default + torchrun), so this fires identically on every rank — no split-hang — and
    # also guards the expert_size_per_partition=E//WORLD arg handed to the factory below.
    assert E % WORLD == 0, (
        f"EP_EXPERTS={E} must be divisible by world size {WORLD} "
        f"({E % WORLD} experts would be orphaned) — pick nodes×8 that divides {E}"
    )

    install_mpi_shim()

    from tensorrt_llm._torch.modules.fused_moe.communication.communication_factory import (
        CommunicationFactory,
    )
    from tensorrt_llm._torch.modules.fused_moe.communication.nccl_ep import NcclEP
    from tensorrt_llm.mapping import Mapping

    # enable_attention_dp + moe_tp_size=1 are what a wide-EP serve runs; without
    # them create_strategy short-circuits to None / AllGatherReduceScatter BEFORE
    # it ever reads TRTLLM_FORCE_COMM_METHOD (same gate serve.sh satisfies via
    # its extra_llm_api_options YAML).
    mapping = Mapping(
        world_size=WORLD,
        rank=RANK,
        gpus_per_node=8,
        tp_size=WORLD,
        moe_tp_size=1,
        moe_ep_size=WORLD,
        enable_attention_dp=True,
    )
    mc = build_model_config(mapping)

    reason = CommunicationFactory._get_nccl_ep_unavailable_reason(
        torch.bfloat16, None, E, H, TOK, None, K
    )
    log(f"PROBE-GATE unavailable_reason={reason!r}")

    # The real serve entrypoint. Reads TRTLLM_FORCE_COMM_METHOD itself.
    strategy = CommunicationFactory.create_strategy(
        model_config=mc,
        num_experts=E,
        num_slots=E,
        top_k=K,
        expert_size_per_partition=E // WORLD,
    )
    log(f"PROBE-SELECTED type={type(strategy).__name__}")
    # Agree the selection verdict across ranks BEFORE any rank returns: the factory's
    # feasibility gate (_get_nccl_ep_unavailable_reason's per-device shared-memory check)
    # and NcclEP.__init__'s is_nccl_ep_installed() are per-rank evaluations that a
    # heterogeneous node or a divergent library path can split. A bare per-rank return
    # would leave the survivors blocked in the bootstrap collectives until torchrun's
    # timeout — same all-reduce(MIN) contract the combine verdict below uses.
    sel_ok = torch.tensor([1 if isinstance(strategy, NcclEP) else 0], device=dev, dtype=torch.int32)
    dist.all_reduce(sel_ok, op=dist.ReduceOp.MIN)
    if int(sel_ok.item()) != 1:
        log(f"PROBE-FAIL factory returned {type(strategy).__name__}, not NcclEP (on this or a peer rank)")
        return 5

    ctx = strategy._get_context()
    # NcclEpContext on 1.3.0rc24 exposes `layout` but NOT `_ep_algorithm` — the algorithm is
    # not stored on the context in the unpatched factory (upstream hardcodes LOW_LATENCY). A
    # getattr-guard keeps the diagnostic from AttributeError-ing on a GIN-working substrate:
    # when the attr is absent, report the requested algo (env), which == what runs on the
    # unpatched default path. Prefer the real attr if a future base exposes it.
    _algo = getattr(ctx, "_ep_algorithm", None)
    algo_name = _algo.name if _algo is not None else os.environ.get("TRTLLM_NCCL_EP_ALGO", "LOW_LATENCY").upper()
    log(f"PROBE-CONFIG algorithm={algo_name} layout={ctx.layout.name}")

    # Upstream default is LOW_LATENCY+RANK_MAJOR; the algorithm knob is only
    # honored on an image built with APPLY_HT_FLAT_PATCH=1 (TensorRT-LLM PR
    # #17715). If HT was explicitly requested, silently probing LL instead
    # would be a false pass — fail loud.
    want = os.environ.get("TRTLLM_NCCL_EP_ALGO", "LOW_LATENCY").upper()
    if want in ("HIGH_THROUGHPUT", "HT"):
        # rc24 does not expose the algorithm as a context attr; layout IS observable
        # (RANK_MAJOR default, FLAT once PR #17715's HT/FLAT patch is baked). Prefer the
        # algo attr if a future base exposes it, else fall back to the layout==FLAT proof.
        algo_ok = (_algo.name == "HIGH_THROUGHPUT") if _algo is not None else (ctx.layout.name == "FLAT")
        if not algo_ok:
            log("PROBE-FAIL env requested HIGH_THROUGHPUT but the factory selected "
                f"{algo_name}+{ctx.layout.name} — unpatched image? "
                "(build with APPLY_HT_FLAT_PATCH=1)")
            return 6

    # Exercise the factory-selected instance cross-node — selection alone does not
    # prove the group works, and the whole defect class here is a first-dispatch IMA.
    g = torch.Generator(device="cpu").manual_seed(4242 + RANK)
    # Snapshot the oracle inputs on CPU FIRST, then hand independent device copies to
    # dispatch. expect (below) is derived from these CPU snapshots, so a kernel that
    # mutated its caller-owned inputs in place could not corrupt the reference and the
    # result identically and still pass — the docstring's "CPU-derived oracle" is now literal.
    x_cpu = (torch.randn(TOK, H, generator=g, dtype=torch.float32) / 8.0).to(torch.bfloat16)
    slots_cpu = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(TOK)]).to(torch.int32)
    scales_cpu = torch.rand(TOK, K, generator=g, dtype=torch.float32)
    x = x_cpu.to(dev)
    slots = slots_cpu.to(dev)
    scales = scales_cpu.to(dev)

    recv_hs, _, recv_slots, recv_scales = strategy.dispatch(
        hidden_states=x,
        hidden_states_sf=None,
        token_selected_slots=slots,
        token_final_scales=scales,
        all_rank_num_tokens=[TOK] * WORLD,
    )
    torch.cuda.synchronize()
    log(f"PROBE-DISPATCH recv_hs={tuple(recv_hs.shape)}")

    # Local "expert compute" stand-in with a closed-form oracle: expert e scales its
    # tokens by (e+1)/E, so the combined output must equal
    # x * sum_k(((slot_k+1)/E) * scale_k) — checkable per token without real weights.
    num_local = E // WORLD
    lo, hi = RANK * num_local, RANK * num_local + num_local
    moe_out = torch.zeros_like(recv_hs, dtype=torch.bfloat16)
    sl = recv_slots.to(torch.int64)
    for kk in range(recv_slots.shape[1]):
        e = sl[:, kk]
        m = (e >= lo) & (e < hi)
        if not bool(m.any()):
            continue
        f = ((e[m].to(torch.float32) + 1.0) / E).unsqueeze(1)
        w = recv_scales[m, kk].unsqueeze(1).to(torch.float32)
        moe_out[m] += (recv_hs[m].to(torch.float32) * f * w).to(torch.bfloat16)

    combined = strategy.combine(moe_out, all_rank_max_num_tokens=TOK)
    torch.cuda.synchronize()

    # Oracle from the CPU snapshots (not the tensors dispatch received), moved to device
    # only for the comparison against the device-resident combined output.
    fac = ((slots_cpu.to(torch.float32) + 1.0) / E) * scales_cpu
    expect = (x_cpu.to(torch.float32) * fac.sum(1, keepdim=True)).to(dev)
    got = combined.to(torch.float32)
    relmax = ((got - expect).abs().max() / expect.abs().max().clamp_min(1e-6)).item()
    nz = int((got.abs().sum(1) > 0).sum())
    ok = (nz == TOK) and (relmax < 0.05)
    log(f"PROBE-COMBINE nonzero={nz}/{TOK} relmax={relmax:.4g} {'OK' if ok else 'MISMATCH'}")

    # Every rank must agree — one silently-corrupted rank is a failed run.
    flag = torch.tensor([1 if ok else 0], device=dev, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    verdict = "PROBE-PASS" if int(flag.item()) == 1 else "PROBE-MISMATCH"
    log(f"{verdict} factory-selected NcclEP {algo_name}+{ctx.layout.name} world={WORLD}")

    strategy.destroy()
    dist.barrier()
    log("PROBE-DONE")
    return 0 if int(flag.item()) == 1 else 4


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print(f"[rank{RANK}] PROBE-EXC", flush=True)
        sys.exit(1)
