#!/usr/bin/env python3
"""Common-boundary DeepEP-compatible dispatch/combine benchmark.

The benchmark deliberately avoids each backend's native timing and byte
accounting.  Every arm receives the same deterministic BF16 input, exact route
indices, and top-k weights.  A CUDA event pair surrounds the complete dispatch
followed by combine operation, and the slowest rank is the iteration latency.

For FP8 dispatch, the timed boundary starts with a BF16 input.  DeepEP V2's
explicit BF16-to-FP8 preparation is therefore inside the timed region, matching
the conversion already performed inside the V1 and UCCL low-latency APIs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


RESULT_PREFIX = "ADAI_FAIR_RESULT "
SCHEMA_VERSION = 1


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""

    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def global_digest(local_digest: str, device: torch.device) -> str:
    digest_tensor = torch.tensor(
        list(bytes.fromhex(local_digest)), dtype=torch.uint8, device=device
    )
    gathered = [torch.empty_like(digest_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, digest_tensor)
    payload = b"".join(bytes(item.cpu().tolist()) for item in gathered)
    return hashlib.sha256(payload).hexdigest()


def make_route(
    rank: int,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Create a balanced, backend-independent route with unique experts/token."""

    stride = 31
    if math.gcd(stride, num_experts) != 1:
        raise ValueError("route stride must be coprime with num_experts")
    global_token = rank * num_tokens + torch.arange(
        num_tokens, dtype=torch.int64, device=device
    )
    slots = torch.arange(top_k, dtype=torch.int64, device=device)
    route = (
        global_token[:, None] * 17 + slots[None, :] * stride + seed
    ) % num_experts
    if top_k > num_experts:
        raise ValueError("top_k cannot exceed num_experts")
    return route.contiguous()


def make_input(
    rank: int, num_tokens: int, hidden: int, device: torch.device
) -> torch.Tensor:
    """Create deterministic, bounded BF16 data without backend RNG state."""

    row = rank * num_tokens + torch.arange(
        num_tokens, dtype=torch.int64, device=device
    )
    column = torch.arange(hidden, dtype=torch.int64, device=device)
    values = (row[:, None] * 17 + column[None, :] * 13 + 19) % 257
    return ((values - 128).to(torch.float32) / 128.0).to(torch.bfloat16)


def normalized_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual64 = actual.double() + 1
    expected64 = expected.double() + 1
    denominator = (actual64.square() + expected64.square()).sum()
    similarity = 2 * (actual64 * expected64).sum() / denominator
    return float((1 - similarity).item())


def logical_payload_bytes_per_rank(
    route: torch.Tensor,
    rank: int,
    world_size: int,
    local_world_size: int,
    hidden: int,
    dispatch_dtype: str,
    num_experts: int,
) -> tuple[int, int, int]:
    """Return all, scale-out, and valid route selections' logical payload bytes.

    Each valid expert assignment contributes its useful dispatch tensor, FP8
    scales when applicable, and one BF16 combine tensor.  Backend metadata is
    excluded.  Scale-out bytes count assignments owned by a different node.
    """

    if num_experts % world_size:
        raise ValueError("num_experts must divide world_size")
    dispatch_bytes = hidden * 2
    if dispatch_dtype == "fp8":
        dispatch_bytes = hidden + math.ceil(hidden / 128) * 4
    elif dispatch_dtype != "bf16":
        raise ValueError(f"unsupported dispatch dtype: {dispatch_dtype}")
    bytes_per_selection = dispatch_bytes + hidden * 2
    valid = route >= 0
    valid_selections = int(valid.sum().item())
    expert_owner = route // (num_experts // world_size)
    source_node = rank // local_world_size
    destination_node = expert_owner // local_world_size
    remote_selections = int(((destination_node != source_node) & valid).sum().item())
    return (
        valid_selections * bytes_per_selection,
        remote_selections * bytes_per_selection,
        valid_selections,
    )


@dataclass
class DispatchState:
    recv_x: Any
    recv_topk_weights: torch.Tensor | None
    handle: Any


class BackendAdapter:
    def __init__(
        self,
        arm: str,
        group: dist.ProcessGroup,
        num_tokens: int,
        hidden: int,
        num_experts: int,
    ) -> None:
        self.arm = arm
        self.group = group
        self.num_tokens = num_tokens
        self.hidden = hidden
        self.num_experts = num_experts
        self.world_size = dist.get_world_size(group)
        self.buffer: Any
        self._cast_back: Any = None
        self._cast_to_fp8: Any = None

        if arm == "uccl":
            sys.path.insert(0, "/opt/uccl/ep/bench")
            from buffer import Buffer  # type: ignore[import-not-found]
            from utils import per_token_cast_back  # type: ignore[import-not-found]

            self._cast_back = per_token_cast_back
            rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
                num_tokens, hidden, self.world_size, num_experts
            )
            self.buffer = Buffer(
                group,
                num_rdma_bytes=rdma_bytes,
                low_latency_mode=True,
                num_qps_per_rank=num_experts // self.world_size,
                allow_nvlink_for_low_latency_mode=True,
                explicitly_destroy=True,
            )
        elif arm == "deepep-v1-nvshmem":
            sys.path.insert(0, "/opt/amazon/deepep/tests")
            sys.path.insert(0, "/opt/amazon/deepep")
            import deep_ep  # type: ignore[import-not-found]
            from utils import per_token_cast_back  # type: ignore[import-not-found]

            self._cast_back = per_token_cast_back
            rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
                num_tokens, hidden, self.world_size, num_experts
            )
            self.buffer = deep_ep.Buffer(
                group,
                num_rdma_bytes=rdma_bytes,
                low_latency_mode=True,
                num_qps_per_rank=num_experts // self.world_size,
                allow_nvlink_for_low_latency_mode=True,
                explicitly_destroy=True,
                allow_mnnvl=False,
            )
        elif arm == "deepep-v2-gin-gda":
            sys.path.insert(0, "/opt/amazon/deepep-v2")
            import deep_ep  # type: ignore[import-not-found]
            from deep_ep.utils.math import (  # type: ignore[import-not-found]
                per_token_cast_back,
                per_token_cast_to_fp8,
            )

            self._cast_back = per_token_cast_back
            self._cast_to_fp8 = per_token_cast_to_fp8
            self.buffer = deep_ep.ElasticBuffer(
                group,
                num_max_tokens_per_rank=num_tokens,
                hidden=hidden,
                deterministic=False,
                allow_hybrid_mode=True,
                allow_multiple_reduction=True,
                prefer_overlap_with_compute=False,
                sl_idx=0,
                num_allocated_qps=0,
                explicitly_destroy=True,
                num_gpu_timeout_secs=180,
                num_cpu_timeout_secs=180,
            )
        else:
            raise ValueError(f"unsupported arm: {arm}")

    @property
    def is_elastic(self) -> bool:
        return self.arm == "deepep-v2-gin-gda"

    def prepare_dispatch_input(self, x: torch.Tensor, dispatch_dtype: str) -> Any:
        if dispatch_dtype == "bf16":
            return x
        if dispatch_dtype != "fp8":
            raise ValueError(f"unsupported dispatch dtype: {dispatch_dtype}")
        if self.is_elastic:
            return self._cast_to_fp8(x)
        return x

    def dispatch(
        self,
        prepared_x: Any,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        dispatch_dtype: str,
    ) -> DispatchState:
        if self.is_elastic:
            recv_x, _, recv_weights, handle, event = self.buffer.dispatch(
                x=prepared_x,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_experts=self.num_experts,
                num_max_tokens_per_rank=self.num_tokens,
                expert_alignment=1,
                async_with_compute_stream=True,
                allocate_on_comm_stream=False,
                do_handle_copy=True,
                do_cpu_sync=True,
            )
            event.current_stream_wait()
            return DispatchState(recv_x, recv_weights, handle)

        recv_x, _, handle, event, _ = self.buffer.low_latency_dispatch(
            prepared_x,
            topk_idx,
            self.num_tokens,
            self.num_experts,
            use_fp8=dispatch_dtype == "fp8",
            async_finish=True,
            return_recv_hook=False,
        )
        event.current_stream_wait()
        return DispatchState(recv_x, None, handle)

    def received_as_bf16(self, recv_x: Any, dispatch_dtype: str) -> torch.Tensor:
        if dispatch_dtype == "bf16":
            return recv_x
        if self.is_elastic:
            return self._cast_back(recv_x[0], recv_x[1])
        fp8, scales = recv_x
        return self._cast_back(
            fp8.view(-1, self.hidden),
            scales.view(-1, self.hidden // 128),
        ).view(fp8.shape)

    def combine(
        self,
        combine_input: torch.Tensor,
        state: DispatchState,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_elastic:
            combined, _, event = self.buffer.combine(
                x=combine_input,
                handle=state.handle,
                topk_weights=state.recv_topk_weights,
                async_with_compute_stream=True,
                allocate_on_comm_stream=False,
            )
            event.current_stream_wait()
            return combined

        combined, event, _ = self.buffer.low_latency_combine(
            combine_input,
            topk_idx,
            topk_weights,
            state.handle,
            use_logfmt=False,
            async_finish=True,
            return_recv_hook=False,
        )
        event.current_stream_wait()
        return combined

    def destroy(self) -> None:
        self.buffer.destroy()


def initialize_distributed() -> tuple[int, int, int, torch.device, dist.ProcessGroup]:
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    world_size = dist.get_world_size()
    group = dist.new_group(list(range(world_size)))
    torch.set_default_dtype(torch.bfloat16)
    return dist.get_rank(), world_size, local_world_size, device, group


def run_dtype(
    adapter: BackendAdapter,
    x: torch.Tensor,
    route: torch.Tensor,
    topk_weights: torch.Tensor,
    dispatch_dtype: str,
    warmups: int,
    iterations: int,
    rank: int,
    world_size: int,
    local_world_size: int,
    args: argparse.Namespace,
    route_hash: str,
    input_hash: str,
) -> dict[str, Any]:
    prepared = adapter.prepare_dispatch_input(x, dispatch_dtype)
    correctness_state = adapter.dispatch(prepared, route, topk_weights, dispatch_dtype)
    correctness_input = adapter.received_as_bf16(
        correctness_state.recv_x, dispatch_dtype
    )
    correctness_output = adapter.combine(
        correctness_input, correctness_state, route, topk_weights
    )
    torch.cuda.synchronize()
    expected = x * topk_weights.sum(dim=1, keepdim=True).to(torch.bfloat16)
    diff = normalized_diff(correctness_output, expected)
    max_abs_error = float(
        (correctness_output.float() - expected.float()).abs().max().item()
    )
    tolerance = 9e-4 if dispatch_dtype == "fp8" else 1e-5
    correctness_pass = diff <= tolerance and bool(
        torch.isfinite(correctness_output).all().item()
    )
    correctness_tensor = torch.tensor(
        [1 if correctness_pass else 0], dtype=torch.int32, device=x.device
    )
    dist.all_reduce(correctness_tensor, op=dist.ReduceOp.MIN)
    if int(correctness_tensor.item()) != 1:
        raise RuntimeError(
            f"correctness failed for {dispatch_dtype}: diff={diff}, "
            f"tolerance={tolerance}, max_abs_error={max_abs_error}"
        )

    # Timed combine uses a stable, preallocated expert-output tensor.  This
    # keeps expert computation and FP8 dequantization outside the communication
    # boundary while retaining the handle created by each timed dispatch.
    combine_input = torch.zeros_like(correctness_input, dtype=torch.bfloat16)

    def iteration() -> None:
        current_x = adapter.prepare_dispatch_input(x, dispatch_dtype)
        state = adapter.dispatch(current_x, route, topk_weights, dispatch_dtype)
        adapter.combine(combine_input, state, route, topk_weights)

    for _ in range(warmups):
        dist.barrier(group=adapter.group)
        iteration()
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    max_rank_latency_ms: list[float] = []
    for _ in range(iterations):
        dist.barrier(group=adapter.group)
        start.record()
        iteration()
        end.record()
        end.synchronize()
        local_latency = torch.tensor(
            [start.elapsed_time(end)], dtype=torch.float32, device=x.device
        )
        dist.all_reduce(local_latency, op=dist.ReduceOp.MAX, group=adapter.group)
        max_rank_latency_ms.append(float(local_latency.item()))

    logical_bytes, scaleout_bytes, valid_selections = (
        logical_payload_bytes_per_rank(
            route,
            rank,
            world_size,
            local_world_size,
            args.hidden,
            dispatch_dtype,
            args.experts,
        )
    )
    counters = torch.tensor(
        [logical_bytes, scaleout_bytes, valid_selections],
        dtype=torch.int64,
        device=x.device,
    )
    dist.all_reduce(counters, op=dist.ReduceOp.SUM, group=adapter.group)
    avg_logical_bytes = int(counters[0].item()) / world_size
    avg_scaleout_bytes = int(counters[1].item()) / world_size
    global_valid_selections = int(counters[2].item())

    median_ms = statistics.median(max_rank_latency_ms)
    mean_ms = statistics.fmean(max_rank_latency_ms)
    stdev_ms = statistics.stdev(max_rank_latency_ms) if iterations > 1 else 0.0
    result = {
        "schema_version_dimensionless": SCHEMA_VERSION,
        "benchmark": "common-boundary-dispatch-combine",
        "arm": args.arm,
        "run_index_dimensionless": args.run_index,
        "dispatch_dtype": dispatch_dtype,
        "world_size_ranks": world_size,
        "nodes": world_size // local_world_size,
        "gpus_per_node": local_world_size,
        "tokens_per_rank": args.tokens,
        "global_input_tokens": args.tokens * world_size,
        "hidden_dimensions": args.hidden,
        "experts": args.experts,
        "top_k_dimensionless": args.top_k,
        "route_seed_dimensionless": args.seed,
        "route_hash_sha256": route_hash,
        "input_hash_sha256": input_hash,
        "global_valid_expert_selections": global_valid_selections,
        "warmup_iterations": warmups,
        "measured_iterations": iterations,
        "timing_boundary": "BF16 input ready through dispatch and combine completion; slowest rank CUDA elapsed time",
        "logical_payload_definition": "per valid expert assignment: dispatch tensor plus FP8 scales when selected plus BF16 combine tensor; backend metadata excluded",
        "avg_logical_payload_bytes_per_rank": avg_logical_bytes,
        "avg_scaleout_logical_payload_bytes_per_rank": avg_scaleout_bytes,
        "latency_ms": {
            "median": median_ms,
            "mean": mean_ms,
            "p95": percentile(max_rank_latency_ms, 0.95),
            "minimum": min(max_rank_latency_ms),
            "maximum": max(max_rank_latency_ms),
            "stdev": stdev_ms,
            "cv_percent": stdev_ms / mean_ms * 100 if mean_ms else 0.0,
        },
        "aggregate_input_tokens_per_second": (
            args.tokens * world_size / (median_ms / 1e3)
        ),
        "effective_logical_gigabytes_per_second_per_rank": (
            avg_logical_bytes / (median_ms / 1e3) / 1e9
        ),
        "effective_scaleout_logical_gigabytes_per_second_per_rank": (
            avg_scaleout_bytes / (median_ms / 1e3) / 1e9
        ),
        "correctness": {
            "status": "PASS",
            "normalized_diff_dimensionless": diff,
            "tolerance_dimensionless": tolerance,
            "max_abs_error_bf16_value": max_abs_error,
        },
        "runtime": {
            "image_reference": os.environ.get("ADAI_IMAGE_REFERENCE", "unknown"),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": list(torch.cuda.nccl.version()),
            "gpu": torch.cuda.get_device_name(),
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        required=True,
        choices=("uccl", "deepep-v1-nvshmem", "deepep-v2-gin-gda"),
    )
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument(
        "--dispatch-dtypes",
        default="fp8,bf16",
        help="Comma-separated dtype order; allowed values are fp8 and bf16",
    )
    args = parser.parse_args()
    args.dispatch_dtypes = args.dispatch_dtypes.split(",")
    if sorted(args.dispatch_dtypes) != ["bf16", "fp8"]:
        parser.error("--dispatch-dtypes must contain fp8 and bf16 exactly once")
    if args.warmups < 1 or args.iterations < 2:
        parser.error("at least 1 warmup and 2 measured iterations are required")
    return args


def main() -> None:
    args = parse_args()
    rank, world_size, local_world_size, device, group = initialize_distributed()
    if args.experts % world_size:
        raise SystemExit("experts must divide the distributed world size")
    route = make_route(
        rank, args.tokens, args.experts, args.top_k, args.seed, device
    )
    x = make_input(rank, args.tokens, args.hidden, device)
    topk_weights = torch.full(
        (args.tokens, args.top_k),
        1.0 / args.top_k,
        dtype=torch.float32,
        device=device,
    )
    route_hash = global_digest(tensor_sha256(route), device)
    input_hash = global_digest(tensor_sha256(x), device)
    route_histogram = torch.bincount(
        route.flatten(), minlength=args.experts
    ).to(torch.int64)
    dist.all_reduce(route_histogram, op=dist.ReduceOp.SUM, group=group)

    if rank == 0:
        print(
            "ADAI_FAIR_CONFIG "
            + json.dumps(
                {
                    "arm": args.arm,
                    "world_size_ranks": world_size,
                    "tokens_per_rank": args.tokens,
                    "hidden_dimensions": args.hidden,
                    "experts": args.experts,
                    "top_k_dimensionless": args.top_k,
                    "route_hash_sha256": route_hash,
                    "input_hash_sha256": input_hash,
                    "route_histogram_min_selections": int(
                        route_histogram.min().item()
                    ),
                    "route_histogram_max_selections": int(
                        route_histogram.max().item()
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    adapter = BackendAdapter(
        args.arm, group, args.tokens, args.hidden, args.experts
    )
    try:
        for dispatch_dtype in args.dispatch_dtypes:
            result = run_dtype(
                adapter,
                x,
                route,
                topk_weights,
                dispatch_dtype,
                args.warmups,
                args.iterations,
                rank,
                world_size,
                local_world_size,
                args,
                route_hash,
                input_hash,
            )
            if rank == 0:
                print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    finally:
        adapter.destroy()
        dist.barrier(group=group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
