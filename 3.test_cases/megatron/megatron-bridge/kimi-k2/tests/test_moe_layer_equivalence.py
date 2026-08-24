#!/usr/bin/env python3
"""Gate B final MoE output and gradient equivalence against NCCL all-to-all."""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.mappings import all_to_all
from megatron.core.transformer.moe.fused_a2a import (
    destroy_elastic_buffers,
    elastic_fused_combine,
    elastic_fused_dispatch,
)
from megatron.core.transformer.moe.moe_utils import permute, unpermute
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
    experts_per_rank = experts // dist.get_world_size()
    first = dist.get_rank() * experts_per_rank
    valid_recv_indices = recv_indices[recv_indices >= 0]
    assert valid_recv_indices.numel() > 0
    assert int(valid_recv_indices.max()) < experts_per_rank

    # Reproduce _BaseDeepepManager's destination-local multihot conversion and
    # the TE fused permute/unpermute path used by the Kimi configuration. Keeping
    # one row per expert assignment also matches where MCore applies router
    # probabilities before the expert FC2 parameter operation.
    batch_size = recv_indices.shape[0]
    routing_map = torch.zeros(
        (batch_size, experts_per_rank), dtype=torch.bool, device=recv_indices.device
    )
    multihot_probabilities = torch.zeros(
        (batch_size, experts_per_rank),
        dtype=torch.float32,
        device=recv_indices.device,
    )
    mask = recv_indices != -1
    valid_indices = recv_indices[mask]
    row_indices = torch.arange(
        batch_size, device=recv_indices.device
    ).repeat_interleave(mask.sum(dim=1))
    routing_map[row_indices, valid_indices] = True
    multihot_probabilities[row_indices, valid_indices] = recv_probabilities[mask]
    permuted_x, permuted_probabilities, reversed_mapping, pad_offsets, _ = permute(
        recv_x,
        routing_map,
        probs=multihot_probabilities,
        num_out_tokens=int(counts.sum()),
        fused=True,
        tokens_per_expert=counts,
    )
    assert pad_offsets is None
    local_experts = torch.arange(
        experts_per_rank, device=recv_x.device, dtype=torch.int64
    ).repeat_interleave(counts.to(device=recv_x.device, dtype=torch.int64))
    assert local_experts.numel() == permuted_x.shape[0]
    global_experts = local_experts + first
    weighted_x = (permuted_x * permuted_probabilities.unsqueeze(1)).to(permuted_x.dtype)
    local_scale = expert_scale.index_select(0, global_experts).to(permuted_x.dtype)
    expert_output = weighted_x * local_scale.unsqueeze(1)
    local = unpermute(
        expert_output,
        reversed_mapping,
        restore_shape=recv_x.shape,
        routing_map=routing_map,
        fused=True,
        pad_offsets=pad_offsets,
    )
    output, _ = elastic_fused_combine(local, handle._mcore_elastic_buffer, handle)
    return output, counts


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


def nccl_moe(inputs, indices, probabilities, expert_scale, experts):
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
    expected_shape = {
        "world_size_ranks": dist.get_world_size(),
        "tokens_per_rank": args.tokens,
        "hidden_units": args.hidden,
        "experts": args.experts,
        "topk": args.topk,
    }
    for name, value in expected_shape.items():
        if name in tolerance_payload:
            assert (
                tolerance_payload[name] == value
            ), f"tolerance {name}={tolerance_payload[name]} does not match test {value}"
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
    x_nccl = x_elastic.detach().clone().requires_grad_(True)
    p_elastic = probabilities.detach().clone().requires_grad_(True)
    p_nccl = probabilities.detach().clone().requires_grad_(True)
    scale_elastic = torch.linspace(
        0.5, 1.5, args.experts, device=device, dtype=torch.float32, requires_grad=True
    )
    scale_nccl = scale_elastic.detach().clone().requires_grad_(True)

    out_elastic, local_counts = elastic_moe(
        x_elastic, indices, p_elastic, scale_elastic, args.experts
    )
    out_nccl = nccl_moe(x_nccl, indices, p_nccl, scale_nccl, args.experts)
    loss_elastic = out_elastic.float().square().mean()
    loss_nccl = out_nccl.float().square().mean()
    loss_elastic.backward()
    loss_nccl.backward()

    local = slice(
        dist.get_rank() * (args.experts // dist.get_world_size()),
        (dist.get_rank() + 1) * (args.experts // dist.get_world_size()),
    )
    errors = {
        "output": max_error(out_elastic, out_nccl),
        "loss": max_error(loss_elastic.reshape(1), loss_nccl.reshape(1)),
        "d_input": max_error(x_elastic.grad, x_nccl.grad),
        "d_router_probability": max_error(p_elastic.grad, p_nccl.grad),
        "d_expert_parameter": max_error(
            scale_elastic.grad[local], scale_nccl.grad[local]
        ),
    }
    passed = all(errors[name] <= float(tolerance[name]) for name in errors)
    updated_elastic = scale_elastic.detach()[local] - 1e-3 * scale_elastic.grad[local]
    updated_nccl = scale_nccl.detach()[local] - 1e-3 * scale_nccl.grad[local]
    errors["optimizer_step"] = max_error(updated_elastic, updated_nccl)
    passed = passed and errors["optimizer_step"] <= float(tolerance["optimizer_step"])
    gathered_hashes = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_hashes, route_hash)
    local_counts_cuda = local_counts.to(device=device, dtype=torch.int64)
    gathered_counts = [
        torch.empty_like(local_counts_cuda) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered_counts, local_counts_cuda)
    global_counts = torch.cat(gathered_counts).to(torch.int64)
    expected_counts = torch.bincount(indices.flatten(), minlength=args.experts).to(
        torch.int64
    )
    dist.all_reduce(expected_counts)
    expected_routes = args.tokens * dist.get_world_size() * args.topk
    exact_route_counts = torch.equal(global_counts, expected_counts)
    no_token_drop = exact_route_counts and int(global_counts.sum()) == expected_routes
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
                    "reference_backend": "nccl-alltoall",
                    "dispatched_index_space": "destination-local-expert",
                    "local_permutation": "mcore-te-fused",
                    "route_hashes": gathered_hashes,
                    "exact_global_expert_token_counts": exact_route_counts,
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
