#!/usr/bin/env python3
"""Validate and summarize common-boundary EP benchmark logs."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PREFIX = "ADAI_FAIR_RESULT "
ARMS = ("uccl", "deepep-v1-nvshmem", "deepep-v2-gin-gda")
WORLD_SIZES = (16, 32)
DTYPES = ("fp8", "bf16")


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_median_ci(
    values: list[float], seed: int, samples: int = 20_000
) -> tuple[float, float]:
    rng = random.Random(seed)
    medians = [
        statistics.median(rng.choices(values, k=len(values))) for _ in range(samples)
    ]
    return percentile(medians, 0.025), percentile(medians, 0.975)


def load_results(root: Path) -> list[dict[str, Any]]:
    results: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    sources: dict[tuple[str, int, int, str], Path] = {}
    for path in sorted(root.rglob("*.log")):
        for line in path.read_text(errors="replace").splitlines():
            if not line.startswith(PREFIX):
                continue
            result = json.loads(line[len(PREFIX) :])
            key = (
                result["arm"],
                result["world_size_ranks"],
                result["run_index_dimensionless"],
                result["dispatch_dtype"],
            )
            if key in results and results[key] != result:
                raise ValueError(
                    f"conflicting result for {key}: {sources[key]} and {path}"
                )
            results[key] = result
            sources[key] = path
    return list(results.values())


def validate(results: list[dict[str, Any]], starts: int) -> None:
    expected = {
        (arm, world, run, dtype)
        for arm in ARMS
        for world in WORLD_SIZES
        for run in range(1, starts + 1)
        for dtype in DTYPES
    }
    observed = {
        (
            result["arm"],
            result["world_size_ranks"],
            result["run_index_dimensionless"],
            result["dispatch_dtype"],
        )
        for result in results
        if result["run_index_dimensionless"] > 0
    }
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise ValueError(f"result matrix mismatch; missing={sorted(missing)}, extra={sorted(extra)}")

    measured = [result for result in results if result["run_index_dimensionless"] > 0]
    for result in measured:
        if result["correctness"]["status"] != "PASS":
            raise ValueError(f"correctness did not pass: {result}")
        if "@sha256:" not in result["runtime"]["image_reference"]:
            raise ValueError(f"image is not digest pinned: {result['runtime']['image_reference']}")

    for world in WORLD_SIZES:
        same_world = [result for result in measured if result["world_size_ranks"] == world]
        route_hashes = {result["route_hash_sha256"] for result in same_world}
        input_hashes = {result["input_hash_sha256"] for result in same_world}
        if len(route_hashes) != 1 or len(input_hashes) != 1:
            raise ValueError(
                f"EP{world} did not replay one route/input: "
                f"routes={route_hashes}, inputs={input_hashes}"
            )


def arm_summary(results: Iterable[dict[str, Any]], seed: int) -> dict[str, Any]:
    ordered = sorted(results, key=lambda item: item["run_index_dimensionless"])
    latencies = [item["latency_ms"]["median"] for item in ordered]
    token_rates = [item["aggregate_input_tokens_per_second"] for item in ordered]
    logical_rates = [
        item["effective_logical_gigabytes_per_second_per_rank"] for item in ordered
    ]
    scaleout_rates = [
        item["effective_scaleout_logical_gigabytes_per_second_per_rank"]
        for item in ordered
    ]
    mean_latency = statistics.fmean(latencies)
    stdev_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    ci_low, ci_high = bootstrap_median_ci(latencies, seed)
    return {
        "starts": len(ordered),
        "run_indices_dimensionless": [
            item["run_index_dimensionless"] for item in ordered
        ],
        "per_start_median_latency_ms": latencies,
        "median_latency_ms": statistics.median(latencies),
        "bootstrap_95_percent_ci_latency_ms": [ci_low, ci_high],
        "run_to_run_cv_percent": (
            stdev_latency / mean_latency * 100 if mean_latency else 0.0
        ),
        "median_aggregate_input_tokens_per_second": statistics.median(token_rates),
        "median_effective_logical_gigabytes_per_second_per_rank": statistics.median(
            logical_rates
        ),
        "median_effective_scaleout_logical_gigabytes_per_second_per_rank": statistics.median(
            scaleout_rates
        ),
    }


def summarize(results: list[dict[str, Any]], starts: int) -> dict[str, Any]:
    measured = [result for result in results if result["run_index_dimensionless"] > 0]
    by_cell_arm: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in measured:
        by_cell_arm[
            (result["world_size_ranks"], result["dispatch_dtype"], result["arm"])
        ].append(result)

    cells = []
    for world in WORLD_SIZES:
        for dtype in DTYPES:
            arms = {
                arm: arm_summary(
                    by_cell_arm[(world, dtype, arm)],
                    seed=20260824 + world + len(dtype) + index,
                )
                for index, arm in enumerate(ARMS)
            }
            comparisons = {}
            v2_by_run = {
                result["run_index_dimensionless"]: result["latency_ms"]["median"]
                for result in by_cell_arm[(world, dtype, "deepep-v2-gin-gda")]
            }
            for index, baseline in enumerate(("uccl", "deepep-v1-nvshmem")):
                baseline_by_run = {
                    result["run_index_dimensionless"]: result["latency_ms"]["median"]
                    for result in by_cell_arm[(world, dtype, baseline)]
                }
                paired = [
                    (baseline_by_run[run] - v2_by_run[run])
                    / baseline_by_run[run]
                    * 100
                    for run in range(1, starts + 1)
                ]
                ci_low, ci_high = bootstrap_median_ci(
                    paired, 20260900 + world + index
                )
                stable = (
                    arms[baseline]["run_to_run_cv_percent"] <= 5
                    and arms["deepep-v2-gin-gda"]["run_to_run_cv_percent"] <= 5
                )
                comparisons[f"deepep-v2-gin-gda_vs_{baseline}"] = {
                    "paired_latency_reduction_percent_per_start": paired,
                    "median_paired_latency_reduction_percent": statistics.median(
                        paired
                    ),
                    "bootstrap_95_percent_ci_reduction_percent": [ci_low, ci_high],
                    "winner_supported": stable and (ci_low > 0 or ci_high < 0),
                }
            same_world = [result for result in measured if result["world_size_ranks"] == world]
            cells.append(
                {
                    "world_size_ranks": world,
                    "dispatch_dtype": dtype,
                    "route_hash_sha256": same_world[0]["route_hash_sha256"],
                    "input_hash_sha256": same_world[0]["input_hash_sha256"],
                    "arms": arms,
                    "comparisons": comparisons,
                }
            )
    return {
        "schema_version_dimensionless": 1,
        "status": "PASS",
        "independent_starts_per_cell": starts,
        "timing_boundary": measured[0]["timing_boundary"],
        "logical_payload_definition": measured[0]["logical_payload_definition"],
        "cells": cells,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Fair Common-Boundary EP Results",
        "",
        f"Each cell has {summary['independent_starts_per_cell']} independent process starts. Latency is the slowest-rank CUDA elapsed time from BF16 input readiness through dispatch and combine completion. Values are medians across independent starts.",
        "",
        "| EP size | Dispatch dtype | Backend | Latency (ms) | 95% bootstrap CI (ms) | Run-to-run CV (%) | Input throughput (tokens/s) | Logical throughput (GB/s/rank) | Scale-out logical throughput (GB/s/rank) |",
        "|---:|:---:|:---|---:|:---:|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        for arm in ARMS:
            value = cell["arms"][arm]
            ci = value["bootstrap_95_percent_ci_latency_ms"]
            lines.append(
                f"| {cell['world_size_ranks']} ranks | {cell['dispatch_dtype'].upper()} | {arm} | "
                f"{value['median_latency_ms']:.4f} ms | [{ci[0]:.4f}, {ci[1]:.4f}] ms | "
                f"{value['run_to_run_cv_percent']:.2f}% | "
                f"{value['median_aggregate_input_tokens_per_second']:.2f} tokens/s | "
                f"{value['median_effective_logical_gigabytes_per_second_per_rank']:.2f} GB/s/rank | "
                f"{value['median_effective_scaleout_logical_gigabytes_per_second_per_rank']:.2f} GB/s/rank |"
            )
    lines.extend(
        [
            "",
            "## Paired DeepEP V2 latency deltas",
            "",
            "Positive values mean DeepEP V2 had lower latency. A winner is supported only when the paired bootstrap interval excludes 0% and both arms have at most 5% run-to-run CV.",
            "",
            "| EP size | Dispatch dtype | Baseline | Median reduction (%) | 95% bootstrap CI (%) | Winner supported |",
            "|---:|:---:|:---|---:|:---:|:---:|",
        ]
    )
    for cell in summary["cells"]:
        for baseline in ("uccl", "deepep-v1-nvshmem"):
            comparison = cell["comparisons"][f"deepep-v2-gin-gda_vs_{baseline}"]
            ci = comparison["bootstrap_95_percent_ci_reduction_percent"]
            lines.append(
                f"| {cell['world_size_ranks']} ranks | {cell['dispatch_dtype'].upper()} | {baseline} | "
                f"{comparison['median_paired_latency_reduction_percent']:.2f}% | "
                f"[{ci[0]:.2f}, {ci[1]:.2f}]% | "
                f"{'yes' if comparison['winner_supported'] else 'no'} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    results = load_results(args.root)
    validate(results, args.starts)
    summary = summarize(results, args.starts)
    args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(markdown(summary))
    print(
        f"PASS fair EP matrix: {len(summary['cells'])} cells, "
        f"{args.starts} independent starts per arm/cell"
    )


if __name__ == "__main__":
    main()
