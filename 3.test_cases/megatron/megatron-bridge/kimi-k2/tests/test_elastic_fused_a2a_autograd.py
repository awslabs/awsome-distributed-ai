#!/usr/bin/env python3
"""Distributed Gate A and multi-flight autograd contract for ElasticBuffer."""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.fused_a2a import elastic_fused_combine, elastic_fused_dispatch


def route(tokens: int, experts: int, topk: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(1234 + dist.get_rank())
    scores = torch.rand((tokens, experts), generator=generator, device=device, dtype=torch.float32)
    weights, indices = torch.topk(scores, topk, dim=1, sorted=False)
    return indices.to(torch.int64), torch.softmax(weights, dim=1)


def one_flight(tokens: int, hidden: int, experts: int, topk: int, asynchronous: bool):
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    indices, probabilities = route(tokens, experts, topk, device)
    probabilities.requires_grad_(True)
    inputs = torch.randn((tokens, hidden), device=device, dtype=torch.bfloat16, requires_grad=True)
    recv_x, recv_indices, recv_probabilities, counts, handle = elastic_fused_dispatch(
        inputs, indices, probabilities, experts, dist.group.WORLD,
        async_finish=asynchronous, allocate_on_comm_stream=asynchronous,
    )
    assert counts.numel() == experts // dist.get_world_size()
    assert recv_indices.shape == recv_probabilities.shape
    combined, _ = elastic_fused_combine(
        recv_x * recv_probabilities.sum(dim=1, keepdim=True).to(recv_x.dtype),
        handle._mcore_elastic_buffer,
        handle,
        async_finish=asynchronous,
        allocate_on_comm_stream=asynchronous,
    )
    loss = combined.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert probabilities.grad is not None and torch.isfinite(probabilities.grad).all()
    return float(loss), handle


def multiple_in_flight(tokens: int, hidden: int, experts: int, topk: int) -> None:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    flights = []
    for offset in (0, 1):
        indices, probabilities = route(tokens + offset, experts, topk, device)
        probabilities.requires_grad_(True)
        inputs = torch.randn((tokens + offset, hidden), device=device, dtype=torch.bfloat16, requires_grad=True)
        recv_x, _, recv_probabilities, _, handle = elastic_fused_dispatch(
            inputs, indices, probabilities, experts, dist.group.WORLD,
            async_finish=True, allocate_on_comm_stream=True,
        )
        flights.append((inputs, probabilities, recv_x, recv_probabilities, handle))
    assert flights[0][4] is not flights[1][4]
    losses = []
    for inputs, probabilities, recv_x, recv_probabilities, handle in reversed(flights):
        output, _ = elastic_fused_combine(
            recv_x * recv_probabilities.sum(dim=1, keepdim=True).to(recv_x.dtype),
            handle._mcore_elastic_buffer, handle,
            async_finish=True, allocate_on_comm_stream=True,
        )
        losses.append(output.float().square().mean())
    sum(losses).backward()
    for inputs, probabilities, *_ in flights:
        assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
        assert probabilities.grad is not None and torch.isfinite(probabilities.grad).all()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert args.experts % dist.get_world_size() == 0
    sync_loss, sync_handle = one_flight(args.tokens, args.hidden, args.experts, args.topk, False)
    async_loss, async_handle = one_flight(args.tokens, args.hidden, args.experts, args.topk, True)
    assert sync_handle is not async_handle
    multiple_in_flight(args.tokens, args.hidden, args.experts, args.topk)
    if dist.get_rank() == 0:
        print("GATE_A_PASS " + json.dumps({
            "world_size_ranks": dist.get_world_size(), "tokens_per_rank": args.tokens,
            "hidden_units": args.hidden, "experts": args.experts, "topk": args.topk,
            "sync_loss": sync_loss, "async_loss": async_loss,
            "cached_handles_distinct": True, "use_fp8_dispatch": False,
            "do_expand": False, "do_cpu_sync": True, "expert_alignment": 1,
            "allow_hybrid_mode": True,
        }, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
