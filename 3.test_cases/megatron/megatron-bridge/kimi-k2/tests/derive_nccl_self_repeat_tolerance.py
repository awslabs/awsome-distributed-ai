#!/usr/bin/env python3
"""Derive Gate B tolerances from a repeated BF16 NCCL all-to-all MoE reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.mappings import all_to_all

METRICS = (
    "output",
    "loss",
    "d_input",
    "d_router_probability",
    "d_expert_parameter",
    "optimizer_step",
)


@dataclass
class Result:
    output: torch.Tensor
    loss: torch.Tensor
    d_input: torch.Tensor
    d_probability: torch.Tensor
    d_scale: torch.Tensor
    updated_scale: torch.Tensor


def route(scores: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities, indices = torch.topk(scores, topk, dim=1, sorted=False)
    return torch.softmax(probabilities, dim=1), indices.to(torch.int64)


def exchange_splits(
    destination: torch.Tensor, world: int
) -> tuple[list[int], list[int]]:
    input_splits_tensor = torch.bincount(destination, minlength=world).to(torch.int64)
    gathered = [torch.empty_like(input_splits_tensor) for _ in range(world)]
    dist.all_gather(gathered, input_splits_tensor)
    rank = dist.get_rank()
    input_splits = [int(value) for value in input_splits_tensor.cpu().tolist()]
    output_splits = [int(counts[rank]) for counts in gathered]
    return input_splits, output_splits


def nccl_moe(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    probabilities: torch.Tensor,
    expert_scale: torch.Tensor,
    experts: int,
) -> torch.Tensor:
    world = dist.get_world_size()
    rank = dist.get_rank()
    experts_per_rank = experts // world
    destination = torch.div(indices, experts_per_rank, rounding_mode="floor").flatten()
    permutation = torch.argsort(destination, stable=True)
    input_splits, output_splits = exchange_splits(destination, world)

    token_ids = (
        torch.arange(inputs.shape[0], device=inputs.device, dtype=torch.int64)
        .unsqueeze(1)
        .expand_as(indices)
        .flatten()
        .index_select(0, permutation)
    )
    send_experts = indices.flatten().index_select(0, permutation).contiguous()
    send_x = inputs.index_select(0, token_ids).contiguous()
    send_probabilities = (
        probabilities.flatten().index_select(0, permutation).contiguous()
    )

    recv_x = all_to_all(dist.group.WORLD, send_x, output_splits, input_splits)
    recv_probabilities = all_to_all(
        dist.group.WORLD, send_probabilities, output_splits, input_splits
    )
    recv_experts = torch.empty(
        (sum(output_splits),), device=inputs.device, dtype=torch.int64
    )
    dist.all_to_all_single(
        recv_experts,
        send_experts,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
    )
    first_expert = rank * experts_per_rank
    last_expert = first_expert + experts_per_rank
    assert bool(((recv_experts >= first_expert) & (recv_experts < last_expert)).all())

    local_scale = expert_scale.index_select(0, recv_experts).to(recv_x.dtype)
    local_output = recv_x * recv_probabilities.unsqueeze(1).to(recv_x.dtype)
    local_output = local_output * local_scale.unsqueeze(1)
    returned = all_to_all(dist.group.WORLD, local_output, input_splits, output_splits)
    return torch.zeros_like(inputs).index_add(0, token_ids, returned)


def execute_nccl(
    base_inputs: torch.Tensor,
    indices: torch.Tensor,
    base_probabilities: torch.Tensor,
    base_scale: torch.Tensor,
    experts: int,
) -> Result:
    inputs = base_inputs.detach().clone().requires_grad_(True)
    probabilities = base_probabilities.detach().clone().requires_grad_(True)
    scale = base_scale.detach().clone().requires_grad_(True)
    output = nccl_moe(inputs, indices, probabilities, scale, experts)
    loss = output.float().square().mean()
    loss.backward()
    local = slice(
        dist.get_rank() * (experts // dist.get_world_size()),
        (dist.get_rank() + 1) * (experts // dist.get_world_size()),
    )
    return Result(
        output.detach(),
        loss.detach(),
        inputs.grad.detach(),
        probabilities.grad.detach(),
        scale.grad.detach()[local],
        scale.detach()[local] - 1e-3 * scale.grad.detach()[local],
    )


def execute_reference(
    base_inputs: torch.Tensor,
    indices: torch.Tensor,
    base_probabilities: torch.Tensor,
    base_scale: torch.Tensor,
    experts: int,
) -> Result:
    inputs = base_inputs.detach().clone().requires_grad_(True)
    probabilities = base_probabilities.detach().clone().requires_grad_(True)
    scale = base_scale.detach().clone().requires_grad_(True)
    coefficient = (probabilities * scale[indices]).sum(dim=1, keepdim=True)
    output = inputs * coefficient.to(inputs.dtype)
    loss = output.float().square().mean()
    loss.backward()
    dist.all_reduce(scale.grad)
    local = slice(
        dist.get_rank() * (experts // dist.get_world_size()),
        (dist.get_rank() + 1) * (experts // dist.get_world_size()),
    )
    return Result(
        output.detach(),
        loss.detach(),
        inputs.grad.detach(),
        probabilities.grad.detach(),
        scale.grad.detach()[local],
        scale.detach()[local] - 1e-3 * scale.grad.detach()[local],
    )


def max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    value = (left.float() - right.float()).abs().max().to(torch.float64)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value)


def errors(left: Result, right: Result) -> dict[str, float]:
    return {
        "output": max_error(left.output, right.output),
        "loss": max_error(left.loss.reshape(1), right.loss.reshape(1)),
        "d_input": max_error(left.d_input, right.d_input),
        "d_router_probability": max_error(left.d_probability, right.d_probability),
        "d_expert_parameter": max_error(left.d_scale, right.d_scale),
        "optimizer_step": max_error(left.updated_scale, right.updated_scale),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, choices=(512, 2048), required=True)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    world = dist.get_world_size()
    assert args.experts % world == 0
    generator = torch.Generator(device=device).manual_seed(20260823 + dist.get_rank())
    scores = torch.randn(
        (args.tokens, args.experts),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    probabilities, indices = route(scores, args.topk)
    base_inputs = torch.randn(
        (args.tokens, args.hidden),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    base_scale = torch.linspace(
        0.5, 1.5, args.experts, device=device, dtype=torch.float32
    )

    first = execute_nccl(base_inputs, indices, probabilities, base_scale, args.experts)
    second = execute_nccl(base_inputs, indices, probabilities, base_scale, args.experts)
    reference = execute_reference(
        base_inputs, indices, probabilities, base_scale, args.experts
    )
    first_reference = errors(first, reference)
    second_reference = errors(second, reference)
    self_repeat = errors(first, second)
    tolerance = {
        name: max(first_reference[name], second_reference[name], self_repeat[name])
        for name in METRICS
    }

    counts = torch.bincount(indices.flatten(), minlength=args.experts).to(torch.int64)
    dist.all_reduce(counts)
    route_digest = hashlib.sha256(indices.cpu().numpy().tobytes()).hexdigest()
    route_digests = [None] * world
    dist.all_gather_object(route_digests, route_digest)
    result = {
        "gate": "NCCL_BF16_SELF_REPEAT",
        "world_size_ranks": world,
        "tokens_per_rank": args.tokens,
        "hidden_units": args.hidden,
        "experts": args.experts,
        "topk": args.topk,
        "first_reference_errors": first_reference,
        "second_reference_errors": second_reference,
        "self_repeat_deltas": self_repeat,
        "tolerance": tolerance,
        "route_hashes": route_digests,
        "global_expert_token_counts": {
            "sum": int(counts.sum()),
            "min": int(counts.min()),
            "max": int(counts.max()),
            "sha256": hashlib.sha256(counts.cpu().numpy().tobytes()).hexdigest(),
        },
    }
    if dist.get_rank() == 0:
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        Path("/run-artifacts/nccl-self-repeat-envelope.json").write_text(payload)
        print("NCCL_SELF_REPEAT_ENVELOPE " + json.dumps(result, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
