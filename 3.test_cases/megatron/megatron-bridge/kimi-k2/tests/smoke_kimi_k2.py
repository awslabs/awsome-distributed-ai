#!/usr/bin/env python3
"""Image admission smoke for literal Kimi-K2 configuration and backend identity."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib


def main() -> None:
    arm = os.environ["EP_ARM"]
    identity = json.loads(pathlib.Path("/opt/benchmark/backend.json").read_text())
    assert identity["ep_arm"] == arm
    benchmark = (
        pathlib.Path(__file__).parents[1] / "benchmarks" / "bench_kimi_k2_pretrain.py"
    )
    spec = importlib.util.spec_from_file_location("bench_kimi_k2_pretrain", benchmark)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    config = module.build_config()
    model = config.model
    assert model.num_layers == 61
    assert model.hidden_size == 7168
    assert model.num_moe_experts == 384
    assert model.moe_router_topk == 8
    assert model.expert_tensor_parallel_size == 1
    assert config.checkpoint.save is None
    assert config.checkpoint.load is None
    assert config.validation.eval_iters == 0
    assert config.validation.eval_interval > config.train.train_iters
    expected = {
        "nccl-alltoall": ("alltoall", None),
        "uccl": ("flex", "deepep"),
        "deepep-v1-nvshmem": ("flex", "deepep"),
        "deepep-v2-gin-gda": ("flex", "deepep_v2"),
    }[arm]
    assert (
        model.moe_token_dispatcher_type,
        model.moe_flex_dispatcher_backend,
    ) == expected
    print(
        "KIMI_K2_SMOKE_PASS "
        + json.dumps(
            {
                "arm": arm,
                "layers": model.num_layers,
                "hidden_units": model.hidden_size,
                "experts": model.num_moe_experts,
                "topk": model.moe_router_topk,
                "dispatcher": expected[0],
                "backend": expected[1],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
