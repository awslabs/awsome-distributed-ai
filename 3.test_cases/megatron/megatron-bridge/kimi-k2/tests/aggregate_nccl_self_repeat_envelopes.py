#!/usr/bin/env python3
"""Aggregate Gate B NCCL BF16 envelopes across independent job starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


METRICS = (
    "output",
    "loss",
    "d_input",
    "d_router_probability",
    "d_expert_parameter",
    "optimizer_step",
)
SHAPE_KEYS = (
    "world_size_ranks",
    "tokens_per_rank",
    "hidden_units",
    "experts",
    "topk",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.inputs) < 2:
        parser.error("at least 2 independent NCCL envelopes are required")

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    shape = {key: payloads[0][key] for key in SHAPE_KEYS}
    routes = payloads[0]["route_hashes"]
    counts = payloads[0]["global_expert_token_counts"]
    for payload in payloads:
        assert payload["gate"] == "NCCL_BF16_SELF_REPEAT"
        assert all(payload[key] == value for key, value in shape.items())
        assert payload["route_hashes"] == routes
        assert payload["global_expert_token_counts"] == counts
        assert all(
            math.isfinite(float(payload["tolerance"][metric]))
            and float(payload["tolerance"][metric]) >= 0.0
            for metric in METRICS
        )

    result = {
        "gate": "NCCL_BF16_SELF_REPEAT_AGGREGATE",
        **shape,
        "independent_job_starts": len(payloads),
        "aggregation": "per-metric maximum across independent NCCL job-start envelopes",
        "tolerance": {
            metric: max(float(payload["tolerance"][metric]) for payload in payloads)
            for metric in METRICS
        },
        "route_hashes": routes,
        "global_expert_token_counts": counts,
        "sources": [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "tolerance": payload["tolerance"],
            }
            for path, payload in zip(args.inputs, payloads, strict=True)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("NCCL_SELF_REPEAT_AGGREGATE " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
