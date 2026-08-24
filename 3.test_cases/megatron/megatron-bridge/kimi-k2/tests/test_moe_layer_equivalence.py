#!/usr/bin/env python3
"""Gate B final MoE output and gradient equivalence against an analytic reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.fused_a2a import (
    destroy_elastic_buffers,
    elastic_fused_combine,
    elastic_fused_dispatch,
)
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction


def replay_route(
    scores: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor, str]:
    RouterReplay.clear_global_router_replay_instances()
    replay = RouterReplay()

    def compute(tensor, count, num_groups=None, group_topk=None):
        del num_groups, group_topk
        return torch.topk(tensor, count, dim=1, sorted=False)

    replay.set_router_replay_action(RouterReplayAction.RECORD)
    probabilities, indices = replay.get_replay_topk(
        scores, topk, default_compute_topk=compute
    )
    recorded = replay.get_recorded_indices().detach().clone()
    replay.set_target_indices(recorded)
    replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    replay_probabilities, replay_indices = replay.get_replay_topk(
        scores, topk, default_compute_topk=compute
    )
    replay.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
    backward_probabilities, backward_indices = replay.get_replay_topk(
        scores, topk, default_compute_topk=compute
    )
    assert torch.equal(indices, replay_indices) and torch.equal(
        indices, backward_indices
    )
    assert torch.equal(probabilities, replay_probabilities) and torch.equal(
        probabilities, backward_probabilities
    )
    digest = hashlib.sha256(recorded.cpu().numpy().tobytes()).hexdigest()
    return probabilities, indices, digest


def elastic_moe(inputs, indices, probabilities, expert_scale, experts):
    recv_x, recv_indices, recv_probabilities, counts, handle = elastic_fused_dispatch(
        inputs,
        indices,
        probabilities,
        experts,
        dist.group.WORLD,
        async_finish=False,
        allocate_on_comm_stream=False,
    )
    first = dist.get_rank() * (experts // dist.get_world_size())
    last = first + experts // dist.get_world_size()
    local = torch.zeros_like(recv_x)
    for expert in range(first, last):
        selected = recv_indices == expert
        coefficient = (recv_probabilities * selected).sum(dim=1, keepdim=True)
        local = local + recv_x * coefficient.to(recv_x.dtype) * expert_scale[expert]
    output, _ = elastic_fused_combine(local, handle._mcore_elastic_buffer, handle)
    return output, counts


def reference_moe(inputs, indices, probabilities, expert_scale):
    factors = expert_scale[indices]
    coefficient = (probabilities * factors).sum(dim=1, keepdim=True)
    return inputs * coefficient.to(inputs.dtype)


def max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    value = (left.float() - right.float()).abs().max().to(torch.float64)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, choices=(512, 2048), required=True)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--tolerance-json", required=True, help="NCCL all-to-all self-repeat envelope"
    )
    args = parser.parse_args()
    tolerance_payload = json.loads(open(args.tolerance_json, encoding="utf-8").read())
    tolerance = tolerance_payload.get("tolerance", tolerance_payload)
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    assert args.experts % dist.get_world_size() == 0
    generator = torch.Generator(device=device).manual_seed(20260823 + dist.get_rank())
    scores = torch.randn(
        (args.tokens, args.experts),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    probabilities, indices, route_hash = replay_route(scores, args.topk)
    probabilities = torch.softmax(probabilities, dim=1)

    x_elastic = torch.randn(
        (args.tokens, args.hidden),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    x_reference = x_elastic.detach().clone().requires_grad_(True)
    p_elastic = probabilities.detach().clone().requires_grad_(True)
    p_reference = probabilities.detach().clone().requires_grad_(True)
    scale_elastic = torch.linspace(
        0.5, 1.5, args.experts, device=device, dtype=torch.float32, requires_grad=True
    )
    scale_reference = scale_elastic.detach().clone().requires_grad_(True)

    out_elastic, local_counts = elastic_moe(
        x_elastic, indices, p_elastic, scale_elastic, args.experts
    )
    out_reference = reference_moe(x_reference, indices, p_reference, scale_reference)
    loss_elastic = out_elastic.float().square().mean()
    loss_reference = out_reference.float().square().mean()
    loss_elastic.backward()
    loss_reference.backward()
    dist.all_reduce(scale_reference.grad)

    local = slice(
        dist.get_rank() * (args.experts // dist.get_world_size()),
        (dist.get_rank() + 1) * (args.experts // dist.get_world_size()),
    )
    errors = {
        "output": max_error(out_elastic, out_reference),
        "loss": max_error(loss_elastic.reshape(1), loss_reference.reshape(1)),
        "d_input": max_error(x_elastic.grad, x_reference.grad),
        "d_router_probability": max_error(p_elastic.grad, p_reference.grad),
        "d_expert_parameter": max_error(
            scale_elastic.grad[local], scale_reference.grad[local]
        ),
    }
    passed = all(errors[name] <= float(tolerance[name]) for name in errors)
    updated_elastic = scale_elastic.detach()[local] - 1e-3 * scale_elastic.grad[local]
    updated_reference = (
        scale_reference.detach()[local] - 1e-3 * scale_reference.grad[local]
    )
    errors["optimizer_step"] = max_error(updated_elastic, updated_reference)
    passed = passed and errors["optimizer_step"] <= float(tolerance["optimizer_step"])
    gathered_hashes = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_hashes, route_hash)
    local_counts_cuda = local_counts.to(device=device, dtype=torch.int64)
    gathered_counts = [
        torch.empty_like(local_counts_cuda) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered_counts, local_counts_cuda)
    global_counts = torch.cat(gathered_counts).to(torch.int64)
    expected_routes = args.tokens * dist.get_world_size() * args.topk
    no_token_drop = int(global_counts.sum()) == expected_routes
    passed = passed and no_token_drop
    if dist.get_rank() == 0:
        print(
            ("GATE_B_PASS " if passed else "GATE_B_FAIL ")
            + json.dumps(
                {
                    "errors": errors,
                    "tolerance": tolerance,
                    "tolerance_source": args.tolerance_json,
                    "world_size_ranks": dist.get_world_size(),
                    "tokens_per_rank": args.tokens,
                    "hidden_units": args.hidden,
                    "experts": args.experts,
                    "topk": args.topk,
                    "route_hashes": gathered_hashes,
                    "no_token_drop": no_token_drop,
                    "global_expert_token_counts": {
                        "sum": int(global_counts.sum()),
                        "expected_sum": expected_routes,
                        "min": int(global_counts.min()),
                        "max": int(global_counts.max()),
                        "sha256": hashlib.sha256(
                            global_counts.cpu().numpy().tobytes()
                        ).hexdigest(),
                    },
                },
                sort_keys=True,
            )
        )
    destroy_elastic_buffers()
    dist.destroy_process_group()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
