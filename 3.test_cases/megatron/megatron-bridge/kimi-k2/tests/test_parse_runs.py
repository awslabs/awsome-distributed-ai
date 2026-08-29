"""Regression tests for durable Megatron benchmark parsing."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_parser():
    path = Path(__file__).parents[2] / "bench" / "parse-runs.py"
    spec = importlib.util.spec_from_file_location("megatron_parse_runs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_dispatcher_identity_allows_concatenated_stdout() -> None:
    parser = _load_parser()
    text = (
        "2026-08-27T00:50:23Z RUNTIME_DISPATCHER_IDENTITY "
        '{"arm":"nccl-alltoall","dispatcher_instances_on_rank":7,'
        '"group_sizes":[32],"manager":null,"moe_layer_instances_on_rank":7}'
        'PIPELINE_P2P_IDENTITY {"batch_p2p_comm":false}'
    )

    assert parser.runtime_dispatcher_identities(text) == [
        {
            "arm": "nccl-alltoall",
            "dispatcher_instances_on_rank": 7,
            "group_sizes": [32],
            "manager": None,
            "moe_layer_instances_on_rank": 7,
        }
    ]
