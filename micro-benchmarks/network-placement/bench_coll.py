# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Time NCCL collectives with torch.distributed, one process per GPU. Launch with torchrun's static rendezvous.

The global rank is node_rank * nproc_per_node + local_rank, so the node rank you give each instance decides which
instances NCCL treats as neighbours in its ring. Rank 0 prints one CSV row per collective and message size:
  coll,bytes,iters,time_us,algbw_GBps,busbw_GBps
time_us is the slowest rank's mean time per call. busbw uses the nccl-tests factors: all_reduce 2(n-1)/n,
all_gather and all_to_all (n-1)/n.
"""
import argparse
import os

import torch
import torch.distributed as dist

p = argparse.ArgumentParser()
p.add_argument("--colls", default="all_reduce,all_gather,all_to_all")
p.add_argument("--min-bytes", type=int, default=8)
p.add_argument("--max-bytes", type=int, default=4 << 30)
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--iters", type=int, default=20)
p.add_argument("--tag", default="")
a = p.parse_args()

dist.init_process_group("nccl")
rank, n = dist.get_rank(), dist.get_world_size()
local = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local)
dev = torch.device("cuda", local)
factor = {"all_reduce": 2 * (n - 1) / n, "all_gather": (n - 1) / n, "all_to_all": (n - 1) / n}


def run(coll, nbytes):
    elems = max(nbytes // 2, n)                       # bf16
    elems -= elems % n
    x = torch.ones(elems, dtype=torch.bfloat16, device=dev)
    if coll == "all_reduce":
        def f():
            dist.all_reduce(x)
    elif coll == "all_gather":
        out, shard = torch.empty(elems, dtype=torch.bfloat16, device=dev), x[: elems // n]

        def f():
            dist.all_gather_into_tensor(out, shard)
    elif coll == "all_to_all":
        out = torch.empty_like(x)

        def f():
            dist.all_to_all_single(out, x)
    else:
        raise SystemExit(f"unknown collective {coll}")
    iters = a.iters if nbytes <= (256 << 20) else max(5, a.iters // 4)
    for _ in range(a.warmup):
        f()
    torch.cuda.synchronize()
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        f()
    end.record()
    torch.cuda.synchronize()
    t = torch.tensor([start.elapsed_time(end) * 1000.0 / iters], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)          # the slowest rank sets the time, as in nccl-tests
    t_us = t.item()
    algbw = elems * 2 / (t_us * 1e-6) / 1e9
    return iters, t_us, algbw, algbw * factor[coll]


if rank == 0:
    print(f"# world={n} tag={a.tag} nccl={torch.cuda.nccl.version()}", flush=True)
    print("coll,bytes,iters,time_us,algbw_GBps,busbw_GBps", flush=True)
for coll in a.colls.split(","):
    nb = a.min_bytes
    while nb <= a.max_bytes:
        it, t_us, alg, bus = run(coll, nb)
        if rank == 0:
            print(f"{coll},{nb},{it},{t_us:.2f},{alg:.3f},{bus:.3f}", flush=True)
        nb *= 4 if nb < (1 << 20) else 2
dist.barrier()
dist.destroy_process_group()
