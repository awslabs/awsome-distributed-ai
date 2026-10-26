# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Time NCCL collectives with torch.distributed, one process per GPU.

Launch with torchrun's static rendezvous. The global rank is
node_rank * nproc_per_node + local_rank, so the node rank you give each
instance decides which instances NCCL treats as neighbours in its ring.
Rank 0 prints one CSV row per collective and message size:
  coll,bytes,iters,time_us,algbw_GBps,busbw_GBps
time_us is the slowest rank's mean time per call. busbw uses the nccl-tests
factors: all_reduce 2(n-1)/n, all_gather and all_to_all (n-1)/n.
"""

import argparse
import os

import torch
import torch.distributed as dist


def parse_args():
    """Command-line options."""
    p = argparse.ArgumentParser()
    p.add_argument("--colls", default="all_reduce,all_gather,all_to_all")
    p.add_argument("--min-bytes", type=int, default=8)
    p.add_argument("--max-bytes", type=int, default=4 << 30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--tag", default="")
    return p.parse_args()


def bus_factor(coll, world):
    """nccl-tests bus-bandwidth factor for a collective."""
    if coll == "all_reduce":
        return 2 * (world - 1) / world
    return (world - 1) / world


def make_call(coll, elems, world, dev):
    """Return a function that runs one collective on a bf16 buffer."""
    x = torch.ones(elems, dtype=torch.bfloat16, device=dev)
    if coll == "all_reduce":
        return lambda: dist.all_reduce(x)
    if coll == "all_gather":
        out = torch.empty(elems, dtype=torch.bfloat16, device=dev)
        shard = x[: elems // world]
        return lambda: dist.all_gather_into_tensor(out, shard)
    if coll == "all_to_all":
        out = torch.empty_like(x)
        return lambda: dist.all_to_all_single(out, x)
    raise SystemExit(f"unknown collective {coll}")


def time_collective(args, coll, nbytes, world, dev):
    """Time one collective at one message size.

    Returns (iterations, slowest rank's us per call, algbw, busbw).
    """
    elems = max(nbytes // 2, world)  # bf16
    elems -= elems % world
    call = make_call(coll, elems, world, dev)
    iters = args.iters if nbytes <= (256 << 20) else max(5, args.iters // 4)
    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        call()
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1000.0 / iters
    elapsed = torch.tensor([us], device=dev)
    # The slowest rank sets the time, as in nccl-tests.
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    per_call_us = elapsed.item()
    algbw = elems * 2 / (per_call_us * 1e-6) / 1e9
    return iters, per_call_us, algbw, algbw * bus_factor(coll, world)


def main():
    """Run every collective over the size sweep; print CSV from rank 0."""
    args = parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    if rank == 0:
        nccl = torch.cuda.nccl.version()
        print(f"# world={world} tag={args.tag} nccl={nccl}", flush=True)
        print("coll,bytes,iters,time_us,algbw_GBps,busbw_GBps", flush=True)
    for coll in args.colls.split(","):
        nbytes = args.min_bytes
        while nbytes <= args.max_bytes:
            iters, us, alg, bus = time_collective(
                args, coll, nbytes, world, dev
            )
            if rank == 0:
                row = f"{coll},{nbytes},{iters},{us:.2f},{alg:.3f},{bus:.3f}"
                print(row, flush=True)
            nbytes *= 4 if nbytes < (1 << 20) else 2
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
