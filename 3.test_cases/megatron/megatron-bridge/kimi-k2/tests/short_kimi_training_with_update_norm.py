#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Short literal Kimi-K2 training gate with sampled optimizer-update telemetry.

The full model is sharded, so materializing a second complete parameter copy is
not practical. This gate deterministically snapshots the first bounded slice of
each rank's local parameter shards before every step and reports the aggregate
post-step update norm. The marker is explicitly a sampled norm, not a full-model
optimizer-update norm.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


def load_benchmark():
    path = Path(
        os.environ.get(
            "KIMI_BENCHMARK_ENTRYPOINT",
            "/opt/benchmark/case/kimi-k2/benchmarks/bench_kimi_k2_pretrain.py",
        )
    )
    spec = importlib.util.spec_from_file_location("adai_kimi_k2_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Kimi-K2 benchmark from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SampledOptimizerUpdate(Callback):
    """Measure a deterministic bounded sample of local parameter updates."""

    def __init__(self, elements_per_rank: int) -> None:
        if elements_per_rank <= 0:
            raise ValueError("elements_per_rank must be positive")
        self.elements_per_rank = elements_per_rank
        self.samples: list[tuple[torch.nn.Parameter, int, torch.Tensor]] = []
        self.layout_sha256 = ""
        self.marker_count = 0

    @staticmethod
    def _step(context: CallbackContext) -> int:
        return int(context.state.train_state.step)

    def on_train_step_start(self, context: CallbackContext) -> None:
        remaining = self.elements_per_rank
        samples: list[tuple[torch.nn.Parameter, int, torch.Tensor]] = []
        layout: list[str] = []
        for chunk_index, model_chunk in enumerate(context.model):
            for name, parameter in sorted(
                model_chunk.named_parameters(), key=lambda item: item[0]
            ):
                if not parameter.requires_grad or parameter.numel() == 0:
                    continue
                take = min(remaining, parameter.numel())
                if take <= 0:
                    break
                snapshot = parameter.detach().reshape(-1)[:take].float().clone()
                samples.append((parameter, take, snapshot))
                layout.append(f"{chunk_index}:{name}:{take}")
                remaining -= take
                if remaining == 0:
                    break
            if remaining == 0:
                break
        if not samples:
            raise RuntimeError("no trainable parameter elements were available to sample")
        self.samples = samples
        self.layout_sha256 = hashlib.sha256("\n".join(layout).encode()).hexdigest()

    def on_train_step_end(self, context: CallbackContext) -> None:
        if not self.samples:
            raise RuntimeError("optimizer-update snapshot is missing")
        device = self.samples[0][2].device
        square_sum = torch.zeros(1, device=device, dtype=torch.float64)
        absolute_sum = torch.zeros(1, device=device, dtype=torch.float64)
        max_absolute = torch.zeros(1, device=device, dtype=torch.float32)
        count = torch.zeros(1, device=device, dtype=torch.int64)
        finite = torch.ones(1, device=device, dtype=torch.int32)
        for parameter, take, snapshot in self.samples:
            update = parameter.detach().reshape(-1)[:take].float() - snapshot
            square_sum += update.square().sum(dtype=torch.float64)
            absolute_sum += update.abs().sum(dtype=torch.float64)
            max_absolute = torch.maximum(max_absolute, update.abs().max().reshape(1))
            count += take
            finite &= torch.isfinite(update).all().to(torch.int32).reshape(1)
        if dist.is_initialized():
            dist.all_reduce(square_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(absolute_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(max_absolute, op=dist.ReduceOp.MAX)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        global_count = int(count.item())
        payload = {
            "aggregation": "all_rank_local_parameter_shard_samples",
            "finite": bool(finite.item()),
            "iteration": self._step(context),
            "layout_sha256_rank_0": self.layout_sha256,
            "losses": {
                name: float(value.detach().float().mean().item())
                for name, value in (context.loss_dict or {}).items()
                if isinstance(value, torch.Tensor)
            },
            "sample_limit_elements_per_rank": self.elements_per_rank,
            "sampled_elements": global_count,
            "skipped_iteration": bool(context.skipped_iter),
            "update_l2_parameter_units": math.sqrt(float(square_sum.item())),
            "update_max_abs_parameter_units": float(max_absolute.item()),
            "update_mean_abs_parameter_units": (
                float(absolute_sum.item()) / global_count if global_count else math.nan
            ),
        }
        if context.grad_norm is not None:
            payload["gradient_norm"] = float(context.grad_norm)
        if not payload["finite"]:
            raise RuntimeError(f"non-finite sampled optimizer update: {payload}")
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("OPTIMIZER_UPDATE_SAMPLE " + json.dumps(payload, sort_keys=True), flush=True)
        self.marker_count += 1
        self.samples = []

    def on_train_end(self, context: CallbackContext) -> None:
        del context
        expected = int(os.environ.get("TRAIN_ITERS", "0"))
        if expected and self.marker_count != expected:
            raise RuntimeError(
                f"optimizer-update marker count {self.marker_count} != train iterations {expected}"
            )


def main() -> None:
    benchmark = load_benchmark()
    config = benchmark.build_config()
    if hasattr(config, "logger") and hasattr(config.logger, "log_params_norm"):
        config.logger.log_params_norm = True
    sample_elements = int(os.environ.get("UPDATE_NORM_SAMPLE_ELEMENTS", "262144"))
    pretrain(
        config=config,
        forward_step_func=forward_step,
        callbacks=[SampledOptimizerUpdate(sample_elements)],
    )


if __name__ == "__main__":
    main()
