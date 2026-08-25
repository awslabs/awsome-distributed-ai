#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""probe_rollout.py — does DeepEP V2's real dispatch/combine move a
rollout-shaped tensor across the EFA node boundary, and does the combined
result match an oracle?

This is the transport gate the NeMo-RL rollout path rides: it constructs the
REAL `deep_ep.ElasticBuffer` (the V2 NCCL-GIN buffer, the class the Megatron
flex dispatcher binds to) on a multi-node NCCL group, dispatches a
rollout-shaped activation tensor with random expert routing, runs a local
"expert compute" stand-in with a closed-form oracle, combines, and checks the
combined output per token — no model weights, no checkpoint download, minutes
not hours. Runs under torchrun (see run-rollout-probe.sh) so dispatch/combine
is exercised ACROSS the node boundary, not just instantiated: the defect class
this exists to catch (e.g. the GIN request-ring overflow that draft PR
deepseek-ai/DeepEP#612 fixes) only fires at the first real cross-node dispatch.

Baseline-image note: upstream DeepEP's SM/QP auto-sizers are EFA-blind at the
pinned SHA (auto-QP overruns aws-ofi-nccl's 128-slot GIN request ring;
`get_rdma_gbs()` reads 0 on EFA) — the exact gaps DeepEP#612 fixes. The probe
therefore passes `num_allocated_qps`/`num_sms`/`num_qps` EXPLICITLY
(EP_NUM_QPS=2 is the value the #612 evidence validated on p5en), which keeps
the unpatched baseline probeable; on a patched image the auto-sizers also work.
"""
import os
import sys
import traceback

import torch
import torch.distributed as dist

RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
LOCAL = int(os.environ["LOCAL_RANK"])

E = int(os.environ.get("EP_EXPERTS", "128"))     # routed experts (Qwen3-30B-A3B shape)
TOK = int(os.environ.get("EP_TOKENS", "128"))    # tokens per rank
H = int(os.environ.get("EP_HIDDEN", "2048"))     # hidden dim
K = int(os.environ.get("EP_TOPK", "8"))          # experts per token
NUM_SMS = int(os.environ.get("EP_NUM_SMS", "8"))
NUM_QPS = int(os.environ.get("EP_NUM_QPS", "2"))


def log(msg):
    print(f"[rank{RANK}] {msg}", flush=True)


def main():
    torch.cuda.set_device(LOCAL)
    dist.init_process_group("nccl", rank=RANK, world_size=WORLD)
    dev = torch.device("cuda", LOCAL)
    assert E % WORLD == 0, f"experts ({E}) must divide by EP size ({WORLD})"

    import deep_ep
    from deep_ep import ElasticBuffer

    log(f"probe start world={WORLD} experts={E} tokens={TOK} hidden={H} topk={K} "
        f"sms={NUM_SMS} qps={NUM_QPS}")

    # Buffer bring-up alone is a gate: it creates the NCCL comm handle and the
    # GIN-backed symmetric buffer across all ranks — if the GIN plugin, the
    # gdrcopy handle, or the EFA provider is broken, it dies HERE with a
    # transport error, before any dispatch.
    buf = ElasticBuffer(
        group=dist.group.WORLD,
        num_max_tokens_per_rank=TOK,
        hidden=H,
        num_topk=K,
        num_allocated_qps=NUM_QPS,   # explicit — see module docstring
        explicitly_destroy=True,
    )
    log(f"PROBE-BUFFER ElasticBuffer up: bytes={buf.num_bytes} "
        f"rdma_ranks={buf.num_rdma_ranks} nvlink_ranks={buf.num_nvlink_ranks}")

    # Rollout-shaped payload with a closed-form oracle: expert e scales its
    # tokens by (e+1)/E, weights folded in during local compute, so the
    # combined output must equal x * sum_k(((idx_k+1)/E) * w_k) per token —
    # checkable without real weights.
    idx_dtype = getattr(deep_ep, "topk_idx_t", torch.int64)
    g = torch.Generator(device="cpu").manual_seed(4242 + RANK)
    x = (torch.randn(TOK, H, generator=g, dtype=torch.float32) / 8.0).to(dev, torch.bfloat16)
    topk_idx = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(TOK)]).to(dev, idx_dtype)
    # keep weights away from 0 so no oracle row degenerates to all-zeros
    topk_weights = (torch.rand(TOK, K, generator=g, dtype=torch.float32) * 0.9 + 0.1).to(dev)

    recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buf.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=E,
        num_sms=NUM_SMS,
        num_qps=NUM_QPS,
    )
    torch.cuda.synchronize()
    log(f"PROBE-DISPATCH recv_x={tuple(recv_x.shape)}")

    # Local expert compute against the oracle. Received routing ids are global
    # expert indices (invalid/non-local slots < 0); if a future DeepEP maps
    # them to local ids instead, detect and offset rather than misattribute.
    num_local = E // WORLD
    lo, hi = RANK * num_local, RANK * num_local + num_local
    sl = recv_topk_idx.to(torch.int64)
    if sl.numel() > 0 and int(sl.max()) < num_local and E > num_local:
        log("PROBE-NOTE recv_topk_idx looks local-mapped; offsetting by rank base")
        sl = torch.where(sl >= 0, sl + lo, sl)
    moe_out = torch.zeros_like(recv_x, dtype=torch.bfloat16)
    for kk in range(sl.shape[1]):
        e = sl[:, kk]
        m = (e >= lo) & (e < hi)
        if not bool(m.any()):
            continue
        f = ((e[m].to(torch.float32) + 1.0) / E).unsqueeze(1)
        w = recv_topk_weights[m, kk].unsqueeze(1).to(torch.float32)
        moe_out[m] += (recv_x[m].to(torch.float32) * f * w).to(torch.bfloat16)

    # weights already folded into moe_out — combine just reduces
    combined_x, _, _ = buf.combine(moe_out, handle, num_qps=NUM_QPS)
    torch.cuda.synchronize()

    fac = ((topk_idx.to(torch.float32) + 1.0) / E) * topk_weights
    expect = x.to(torch.float32) * fac.sum(1, keepdim=True)
    got = combined_x.to(torch.float32)
    relmax = ((got - expect).abs().max() / expect.abs().max().clamp_min(1e-6)).item()
    nz = int((got.abs().sum(1) > 0).sum())
    ok = (nz == TOK) and (relmax < 0.05)
    log(f"PROBE-COMBINE nonzero={nz}/{TOK} relmax={relmax:.4g} {'OK' if ok else 'MISMATCH'}")

    # Every rank must agree — one silently-corrupted rank is a failed run.
    flag = torch.tensor([1 if ok else 0], device=dev, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    verdict = "PROBE-PASS" if int(flag.item()) == 1 else "PROBE-MISMATCH"
    log(f"{verdict} ElasticBuffer dispatch/combine world={WORLD}")

    buf.destroy()
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
