#!/usr/bin/env python3
"""Validate and summarize common-boundary EP benchmark logs."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from fair_result_io import load_result_log


ARMS = ("uccl", "deepep-v1-nvshmem", "deepep-v2-gin-gda")
WORLD_SIZES = (16, 32)
DTYPES = ("fp8", "bf16")
ARM_LABELS = {
    "uccl": "UCCL",
    "deepep-v1-nvshmem": "DeepEP V1 NVSHMEM",
    "deepep-v2-gin-gda": "DeepEP V2 NCCL GIN",
}
BOOTSTRAP_SAMPLES = 20_000
MAX_RUN_TO_RUN_CV_PERCENT = 5.0
EXPECTED_STARTS = 3
EXPECTED_WARMUPS = 20
EXPECTED_ITERATIONS = 100
TIMING_BOUNDARY = (
    "BF16 input ready through dispatch and combine completion; "
    "slowest rank CUDA elapsed time"
)
LOGICAL_PAYLOAD_DEFINITION = (
    "per valid expert assignment: dispatch tensor plus FP8 scales when selected "
    "plus BF16 combine tensor; backend metadata excluded"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


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
    values: list[float], seed: int, samples: int = BOOTSTRAP_SAMPLES
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
        for result in load_result_log(path):
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
    if starts != EXPECTED_STARTS:
        raise ValueError(f"scored matrix requires exactly {EXPECTED_STARTS} starts")
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
        raise ValueError(
            f"result matrix mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )

    measured = [result for result in results if result["run_index_dimensionless"] > 0]
    if len(measured) != len(expected):
        raise ValueError(
            f"result matrix has {len(measured)} scored records; expected {len(expected)}"
        )
    expected_shape = {
        "benchmark": "common-boundary-dispatch-combine",
        "tokens_per_rank": 128,
        "hidden_dimensions": 7168,
        "experts": 256,
        "top_k_dimensionless": 8,
        "gpus_per_node": 8,
        "warmup_iterations": EXPECTED_WARMUPS,
        "measured_iterations": EXPECTED_ITERATIONS,
        "timing_boundary": TIMING_BOUNDARY,
        "logical_payload_definition": LOGICAL_PAYLOAD_DEFINITION,
    }
    for result in measured:
        for field, expected_value in expected_shape.items():
            if result.get(field) != expected_value:
                raise ValueError(
                    f"unexpected {field}: expected {expected_value}, got {result.get(field)}"
                )
        world_size = result["world_size_ranks"]
        if result.get("nodes") != world_size // 8:
            raise ValueError(f"invalid node count for EP{world_size}: {result}")
        if result.get("global_input_tokens") != 128 * world_size:
            raise ValueError(f"invalid global input token count: {result}")
        if result["correctness"]["status"] != "PASS":
            raise ValueError(f"correctness did not pass: {result}")
        tolerance = 9e-4 if result["dispatch_dtype"] == "fp8" else 1e-5
        correctness = result["correctness"]
        if correctness.get("tolerance_dimensionless") != tolerance:
            raise ValueError(f"unexpected correctness tolerance: {result}")
        if not math.isfinite(
            correctness.get("normalized_diff_dimensionless", math.inf)
        ):
            raise ValueError(f"non-finite correctness result: {result}")
        if correctness["normalized_diff_dimensionless"] > tolerance:
            raise ValueError(f"correctness exceeds tolerance: {result}")
        image_reference = result["runtime"]["image_reference"]
        digest = image_reference.rsplit("@sha256:", 1)[-1]
        if "@sha256:" not in image_reference or not SHA256_PATTERN.fullmatch(digest):
            raise ValueError(
                f"image is not digest pinned: {result['runtime']['image_reference']}"
            )
        for field in ("gpu", "torch_version", "cuda_version", "nccl_version"):
            if not result["runtime"].get(field):
                raise ValueError(f"runtime is missing {field}: {result}")
        for field in ("route_hash_sha256", "input_hash_sha256"):
            if not SHA256_PATTERN.fullmatch(result.get(field, "")):
                raise ValueError(f"invalid {field}: {result}")
        expected_selections = result["global_input_tokens"] * 8
        if result.get("global_valid_expert_selections") != expected_selections:
            raise ValueError(f"invalid valid-expert selection count: {result}")
        dispatch_bytes = 7_168 * 2
        if result["dispatch_dtype"] == "fp8":
            dispatch_bytes = 7_168 + math.ceil(7_168 / 128) * 4
        expected_logical_bytes = 128 * 8 * (dispatch_bytes + 7_168 * 2)
        logical_bytes = result.get("avg_logical_payload_bytes_per_rank")
        scaleout_bytes = result.get("avg_scaleout_logical_payload_bytes_per_rank")
        if logical_bytes != expected_logical_bytes:
            raise ValueError(f"invalid logical payload: {result}")
        if not isinstance(scaleout_bytes, (int, float)) or not (
            0 < scaleout_bytes <= logical_bytes
        ):
            raise ValueError(f"invalid scale-out logical payload: {result}")
        latency = result.get("latency_ms", {}).get("median", 0)
        if (
            not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency <= 0
        ):
            raise ValueError(f"invalid median latency: {result}")
        elapsed_seconds = latency / 1e3
        expected_metrics = {
            "aggregate_input_tokens_per_second": result["global_input_tokens"]
            / elapsed_seconds,
            "effective_logical_gigabytes_per_second_per_rank": logical_bytes
            / elapsed_seconds
            / 1e9,
            "effective_scaleout_logical_gigabytes_per_second_per_rank": scaleout_bytes
            / elapsed_seconds
            / 1e9,
        }
        for field in (
            "aggregate_input_tokens_per_second",
            "effective_logical_gigabytes_per_second_per_rank",
            "effective_scaleout_logical_gigabytes_per_second_per_rank",
        ):
            value = result.get(field, 0)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"invalid positive metric {field}: {result}")
            if not math.isclose(
                value, expected_metrics[field], rel_tol=1e-12, abs_tol=1e-9
            ):
                raise ValueError(
                    f"metric {field} does not match common accounting: {result}"
                )

    common_fields = (
        "warmup_iterations",
        "measured_iterations",
        "timing_boundary",
        "logical_payload_definition",
    )
    for field in common_fields:
        values = {json.dumps(result[field], sort_keys=True) for result in measured}
        if len(values) != 1:
            raise ValueError(f"scored results disagree on {field}: {sorted(values)}")

    runtime_signatures = {
        json.dumps(
            {
                key: result["runtime"].get(key)
                for key in ("gpu", "torch_version", "cuda_version", "nccl_version")
            },
            sort_keys=True,
        )
        for result in measured
    }
    if len(runtime_signatures) != 1:
        raise ValueError(f"runtime stack mismatch: {sorted(runtime_signatures)}")

    for arm in ARMS:
        image_references = {
            result["runtime"]["image_reference"]
            for result in measured
            if result["arm"] == arm
        }
        if len(image_references) != 1:
            raise ValueError(
                f"{arm} did not use one immutable image: {image_references}"
            )

    for world in WORLD_SIZES:
        same_world = [
            result for result in measured if result["world_size_ranks"] == world
        ]
        route_hashes = {result["route_hash_sha256"] for result in same_world}
        input_hashes = {result["input_hash_sha256"] for result in same_world}
        if len(route_hashes) != 1 or len(input_hashes) != 1:
            raise ValueError(
                f"EP{world} did not replay one route/input: "
                f"routes={route_hashes}, inputs={input_hashes}"
            )
        for dtype in DTYPES:
            same_cell = [
                result for result in same_world if result["dispatch_dtype"] == dtype
            ]
            for field in (
                "avg_logical_payload_bytes_per_rank",
                "avg_scaleout_logical_payload_bytes_per_rank",
                "global_valid_expert_selections",
            ):
                values = {result[field] for result in same_cell}
                if len(values) != 1:
                    raise ValueError(
                        f"EP{world} {dtype} disagrees on {field}: {sorted(values)}"
                    )


def validate_provenance(
    provenance: dict[str, Any], results: list[dict[str, Any]], starts: int
) -> None:
    measured = [result for result in results if result["run_index_dimensionless"] > 0]
    expected_images = {
        arm: next(
            result["runtime"]["image_reference"]
            for result in measured
            if result["arm"] == arm
        )
        for arm in ARMS
    }
    if provenance.get("images") != expected_images:
        raise ValueError(
            f"provenance images do not match scored results: {provenance.get('images')}"
        )
    expected_comparison = {
        "tokens_per_rank": 128,
        "hidden_dimensions": 7_168,
        "experts": 256,
        "top_k_dimensionless": 8,
        "warmup_iterations": EXPECTED_WARMUPS,
        "measured_iterations": EXPECTED_ITERATIONS,
        "independent_starts": starts,
    }
    if provenance.get("comparison") != expected_comparison:
        raise ValueError(
            "provenance comparison does not match the scored matrix: "
            f"{provenance.get('comparison')}"
        )
    for field in ("campaign_id", "created_at_utc", "region", "cluster", "git_commit"):
        if not provenance.get(field):
            raise ValueError(f"provenance is missing {field}")


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


def summarize(
    results: list[dict[str, Any]],
    starts: int,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
                    (baseline_by_run[run] - v2_by_run[run]) / baseline_by_run[run] * 100
                    for run in range(1, starts + 1)
                ]
                ci_low, ci_high = bootstrap_median_ci(paired, 20260900 + world + index)
                stable = (
                    arms[baseline]["run_to_run_cv_percent"] <= MAX_RUN_TO_RUN_CV_PERCENT
                    and arms["deepep-v2-gin-gda"]["run_to_run_cv_percent"]
                    <= MAX_RUN_TO_RUN_CV_PERCENT
                )
                comparisons[f"deepep-v2-gin-gda_vs_{baseline}"] = {
                    "paired_latency_reduction_percent_per_start": paired,
                    "median_paired_latency_reduction_percent": statistics.median(
                        paired
                    ),
                    "bootstrap_95_percent_ci_reduction_percent": [ci_low, ci_high],
                    "direction_supported": stable and (ci_low > 0 or ci_high < 0),
                }
            same_world = [
                result for result in measured if result["world_size_ranks"] == world
            ]
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
    images = {
        arm: next(
            result["runtime"]["image_reference"]
            for result in measured
            if result["arm"] == arm
        )
        for arm in ARMS
    }
    runtime = {
        field: measured[0]["runtime"][field]
        for field in ("gpu", "torch_version", "cuda_version", "nccl_version")
    }
    summary = {
        "schema_version_dimensionless": 1,
        "status": "PASS",
        "scored_result_records_dimensionless": len(measured),
        "independent_starts_per_cell": starts,
        "bootstrap_samples_dimensionless": BOOTSTRAP_SAMPLES,
        "maximum_run_to_run_cv_percent_for_direction_support": MAX_RUN_TO_RUN_CV_PERCENT,
        "timing_boundary": measured[0]["timing_boundary"],
        "logical_payload_definition": measured[0]["logical_payload_definition"],
        "comparison_scope": "synthetic decode dispatch-plus-combine communication workload; not end-to-end training or serving",
        "configuration": {
            "world_sizes_ranks": list(WORLD_SIZES),
            "dispatch_dtypes": list(DTYPES),
            "tokens_per_rank": 128,
            "hidden_dimensions": 7_168,
            "experts_dimensionless": 256,
            "top_k_dimensionless": 8,
            "warmup_iterations_dimensionless": EXPECTED_WARMUPS,
            "measured_iterations_dimensionless": EXPECTED_ITERATIONS,
        },
        "runtime": runtime,
        "images": images,
        "cells": cells,
    }
    if provenance is not None:
        summary["campaign_provenance"] = provenance
    return summary


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Common-Boundary EP Results",
        "",
        f"Each cell has {summary['independent_starts_per_cell']} independent process starts. Latency is the slowest-rank CUDA elapsed time from BF16 input readiness through dispatch and combine completion. Values are medians across independent starts. This is a synthetic communication workload, not an end-to-end training or serving result.",
        "",
        "| EP size | Dispatch dtype | Backend | Latency (ms) | 95% bootstrap CI (ms) | Run-to-run CV (%) | Input throughput (tokens/s) | Logical throughput (GB/s/rank) | Scale-out logical throughput (GB/s/rank) |",
        "|---:|:---:|:---|---:|:---:|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        for arm in ARMS:
            value = cell["arms"][arm]
            ci = value["bootstrap_95_percent_ci_latency_ms"]
            lines.append(
                f"| {cell['world_size_ranks']} ranks | {cell['dispatch_dtype'].upper()} | {ARM_LABELS[arm]} | "
                f"{value['median_latency_ms']:.4f} ms | [{ci[0]:.4f}, {ci[1]:.4f}] ms | "
                f"{value['run_to_run_cv_percent']:.2f}% | "
                f"{value['median_aggregate_input_tokens_per_second']:,.2f} tokens/s | "
                f"{value['median_effective_logical_gigabytes_per_second_per_rank']:.2f} GB/s/rank | "
                f"{value['median_effective_scaleout_logical_gigabytes_per_second_per_rank']:.2f} GB/s/rank |"
            )
    lines.extend(
        [
            "",
            "## Paired DeepEP V2 latency deltas",
            "",
            "Positive values mean DeepEP V2 had lower latency. A direction is supported for this workload only when the paired bootstrap interval excludes 0% and both arms have at most 5% run-to-run CV.",
            "",
            "| EP size | Dispatch dtype | Baseline | Median reduction (%) | 95% bootstrap CI (%) | Direction supported |",
            "|---:|:---:|:---|---:|:---:|:---:|",
        ]
    )
    for cell in summary["cells"]:
        for baseline in ("uccl", "deepep-v1-nvshmem"):
            comparison = cell["comparisons"][f"deepep-v2-gin-gda_vs_{baseline}"]
            ci = comparison["bootstrap_95_percent_ci_reduction_percent"]
            lines.append(
                f"| {cell['world_size_ranks']} ranks | {cell['dispatch_dtype'].upper()} | {ARM_LABELS[baseline]} | "
                f"{comparison['median_paired_latency_reduction_percent']:.2f}% | "
                f"[{ci[0]:.2f}, {ci[1]:.2f}]% | "
                f"{'yes' if comparison['direction_supported'] else 'no'} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    results = load_results(args.root)
    validate(results, args.starts)
    provenance = json.loads(args.provenance.read_text())
    validate_provenance(provenance, results, args.starts)
    summary = summarize(results, args.starts, provenance)
    args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(markdown(summary))
    print(
        f"PASS fair EP matrix: {len(summary['cells'])} cells, "
        f"{args.starts} independent starts per arm/cell"
    )


if __name__ == "__main__":
    main()
