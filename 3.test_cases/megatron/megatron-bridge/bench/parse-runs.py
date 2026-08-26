#!/usr/bin/env python3
"""Parse durable four-arm runs and apply backend-specific validity gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path

ITER = re.compile(r"\biteration\s+(\d+)\s*/\s*(\d+)", re.I)
TIME_MS = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.eE+-]+)", re.I)
LOSS = re.compile(r"\blm loss:\s*([0-9.eE+-]+)", re.I)
GRAD = re.compile(r"grad(?:ient)? norm:\s*([0-9.eE+-]+)", re.I)
TFLOPS = re.compile(r"(?:TFLOP/s/GPU|tflops?)[^0-9+-]*([0-9.eE+-]+)", re.I)
LEARNING_RATE = re.compile(r"\blearning rate:\s*([0-9.eE+-]+)", re.I)
PARAMETERS_NORM = re.compile(r"\bparams norm:\s*([0-9.eE+-]+)", re.I)
SKIPPED_ITERATIONS = re.compile(r"\bnumber of skipped iterations:\s*(\d+)", re.I)
NAN_ITERATIONS = re.compile(r"\bnumber of nan iterations:\s*(\d+)", re.I)


def marker(text: str, *patterns: str) -> bool:
    return any(re.search(pattern, text, re.I) is not None for pattern in patterns)


def finite(values: list[float]) -> bool:
    return bool(values) and all(math.isfinite(item) for item in values)


def read_environment(run: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    path = run / "environment.txt"
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                result[key] = value
    return result


def runtime_manifests(run: Path) -> list[dict]:
    found = []
    for path in run.glob("node-*/runtime-manifest.json"):
        try:
            found.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            pass
    return found


def route_summaries(run: Path) -> list[dict]:
    found = []
    for path in run.glob("node-*/route-summary.json"):
        try:
            found.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            pass
    return found


def runtime_dispatcher_identities(text: str) -> list[dict]:
    found = []
    prefix = "RUNTIME_DISPATCHER_IDENTITY "
    for line in text.splitlines():
        if prefix not in line:
            continue
        try:
            found.append(json.loads(line.partition(prefix)[2]))
        except json.JSONDecodeError:
            pass
    return found


def parse_iteration_records(text: str) -> list[dict[str, float | int]]:
    """Extract one normalized metric record for each optimizer iteration."""

    records: dict[int, dict[str, float | int]] = {}
    for line in text.splitlines():
        match = ITER.search(line)
        if not match:
            continue
        iteration = int(match.group(1))
        record = records.setdefault(
            iteration,
            {
                "iteration_dimensionless": iteration,
                "total_iterations_dimensionless": int(match.group(2)),
            },
        )
        for name, regex in (
            ("elapsed_time_ms", TIME_MS),
            ("lm_loss_dimensionless", LOSS),
            ("gradient_norm_parameter_gradient_units", GRAD),
            ("parameters_norm_parameter_units", PARAMETERS_NORM),
            ("learning_rate_dimensionless", LEARNING_RATE),
            ("tflops_per_gpu", TFLOPS),
        ):
            value = regex.search(line)
            if value:
                record[name] = float(value.group(1))
        for name, regex in (
            ("skipped_iterations_count", SKIPPED_ITERATIONS),
            ("nan_iterations_count", NAN_ITERATIONS),
        ):
            value = regex.search(line)
            if value:
                record[name] = int(value.group(1))
    return [records[iteration] for iteration in sorted(records)]


def write_iteration_artifacts(
    run: Path, records: list[dict[str, float | int]]
) -> dict[str, dict[str, str | int]]:
    columns = [
        "iteration_dimensionless",
        "total_iterations_dimensionless",
        "elapsed_time_ms",
        "lm_loss_dimensionless",
        "gradient_norm_parameter_gradient_units",
        "parameters_norm_parameter_units",
        "learning_rate_dimensionless",
        "tflops_per_gpu",
        "skipped_iterations_count",
        "nan_iterations_count",
    ]
    iteration_path = run / "iteration_metrics.csv"
    with iteration_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    loss_path = run / "loss_curve.csv"
    with loss_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["iteration_dimensionless", "lm_loss_dimensionless"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)

    return {
        "iteration_metrics": {
            "path": str(iteration_path),
            "records": len(records),
            "sha256": hashlib.sha256(iteration_path.read_bytes()).hexdigest(),
        },
        "loss_curve": {
            "path": str(loss_path),
            "records": len(records),
            "sha256": hashlib.sha256(loss_path.read_bytes()).hexdigest(),
        },
    }


def parse_run(run: Path, warmup: int) -> dict:
    env = read_environment(run)
    arm = env.get("ep_arm", run.name)
    log_paths = sorted(run.glob("pod-logs/node-rank-*.log"))
    text = "\n".join(path.read_text(errors="replace") for path in log_paths)
    ordered = parse_iteration_records(text)
    iteration_artifacts = write_iteration_artifacts(run, ordered)
    steady = [
        record for record in ordered if record["iteration_dimensionless"] > warmup
    ]
    times = [
        record["elapsed_time_ms"] for record in steady if "elapsed_time_ms" in record
    ]
    losses = [
        record["lm_loss_dimensionless"]
        for record in ordered
        if "lm_loss_dimensionless" in record
    ]
    gradients = [
        record["gradient_norm_parameter_gradient_units"]
        for record in ordered
        if "gradient_norm_parameter_gradient_units" in record
    ]
    tflops = [
        record["tflops_per_gpu"] for record in steady if "tflops_per_gpu" in record
    ]
    manifests = runtime_manifests(run)
    routes = route_summaries(run)
    dispatcher_identities = runtime_dispatcher_identities(text)
    expected_nodes = len([node for node in env.get("nodes", "").split(",") if node])
    expected_iterations = int(env.get("train_iterations", "0"))
    expected_iteration_numbers = list(range(1, expected_iterations + 1))
    observed_iteration_numbers = [
        int(record["iteration_dimensionless"]) for record in ordered
    ]
    required_iteration_fields = {
        "elapsed_time_ms",
        "lm_loss_dimensionless",
        "gradient_norm_parameter_gradient_units",
        "learning_rate_dimensionless",
        "skipped_iterations_count",
        "nan_iterations_count",
    }
    iteration_metrics_complete = (
        expected_iterations > 0
        and observed_iteration_numbers == expected_iteration_numbers
        and all(required_iteration_fields <= record.keys() for record in ordered)
        and all(
            record["total_iterations_dimensionless"] == expected_iterations
            for record in ordered
        )
    )
    learning_rates = [
        record["learning_rate_dimensionless"]
        for record in ordered
        if "learning_rate_dimensionless" in record
    ]
    skipped_iterations = [
        record["skipped_iterations_count"]
        for record in ordered
        if "skipped_iterations_count" in record
    ]
    nan_iterations = [
        record["nan_iterations_count"]
        for record in ordered
        if "nan_iterations_count" in record
    ]
    manifests_complete = expected_nodes > 0 and len(manifests) == expected_nodes
    routes_complete = expected_nodes > 0 and len(routes) == expected_nodes

    identity = manifests_complete and all(
        item.get("ep_arm") == arm
        and item.get("backend_identity", {}).get("ep_arm") == arm
        for item in manifests
    )
    single_nccl = manifests_complete and all(
        item.get("single_nccl") is True for item in manifests
    )
    build_runtime = manifests_complete and all(
        item.get("nccl_build_runtime_match") is True for item in manifests
    )
    efa_manifest = manifests_complete and all(
        "provider: efa" in item.get("efa_devices", "").lower() for item in manifests
    )
    source_path = run / "manifests" / "run-entrypoint.py"
    source_hash = env.get("run_entrypoint_source_sha256", "")
    source_matches = (
        re.fullmatch(r"[0-9a-f]{64}", source_hash) is not None
        and source_path.is_file()
        and hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
    )
    expected_manager = {
        "nccl-alltoall": None,
        "uccl": "_DeepepManager",
        "deepep-v1-nvshmem": "_DeepepManager",
        "deepep-v2-gin-gda": "_DeepepV2Manager",
    }.get(arm)
    expected_ep = int(env.get("ep", "0"))
    runtime_dispatcher_identity = any(
        item.get("arm") == arm
        and item.get("manager") == expected_manager
        and item.get("group_sizes") == [expected_ep]
        and item.get("dispatcher_instances_on_rank", 0) > 0
        and item.get("dispatcher_instances_on_rank")
        == item.get("moe_layer_instances_on_rank")
        for item in dispatcher_identities
    )
    validity = {
        "entrypoint_source": source_matches,
        "image_identity": identity
        and marker(text, r"EP_IMAGE_IDENTITY_OK|EP_BACKEND_REQUEST"),
        "runtime_dispatcher_identity": runtime_dispatcher_identity,
        "elastic_buffer": marker(
            text, r"buffer=ElasticBuffer|DEEPEP_V2_IMPORT_OK ElasticBuffer=true"
        ),
        "deepep_v2_manager": marker(text, r"manager=_DeepepV2Manager"),
        "single_nccl": single_nccl,
        "nccl_build_runtime_match": build_runtime,
        "gin_type_5": manifests_complete
        and all(
            item.get("environment", {}).get("NCCL_GIN_TYPE") == "5"
            for item in manifests
        ),
        "gdaki_context": marker(
            text, r"GIN GDAKI:\s*createContext done|GDAKI.*createContext.*done"
        ),
        "efa": efa_manifest
        and marker(text, r"Selected provider is efa|NET/OFI.*efa|provider: efa"),
        "no_token_drop": (
            marker(
                text, r"NO_TOKEN_DROP_CONFIG capacity_factor=None token_dropping=False"
            )
            and routes_complete
            and all(
                item.get("records", 0) > 0 and item.get("no_token_drop") is True
                for item in routes
            )
        ),
        "steady_timing": finite(times),
        "iteration_metrics_complete": iteration_metrics_complete,
        "finite_loss": finite(losses),
        "finite_gradient": finite(gradients),
        "finite_learning_rate": finite(learning_rates),
        "no_skipped_iterations": bool(skipped_iterations)
        and max(skipped_iterations) == 0,
        "no_nan_iterations": bool(nan_iterations) and max(nan_iterations) == 0,
    }
    required = [
        "entrypoint_source",
        "image_identity",
        "runtime_dispatcher_identity",
        "single_nccl",
        "nccl_build_runtime_match",
        "efa",
        "no_token_drop",
        "steady_timing",
        "iteration_metrics_complete",
        "finite_loss",
        "finite_gradient",
        "finite_learning_rate",
        "no_skipped_iterations",
        "no_nan_iterations",
    ]
    if arm == "deepep-v2-gin-gda":
        required.extend(
            ["elastic_buffer", "deepep_v2_manager", "gin_type_5", "gdaki_context"]
        )
    status_file = (
        (run / "STATUS").read_text(errors="replace")
        if (run / "STATUS").exists()
        else ""
    )
    complete = status_file.startswith("PASS")
    status = (
        "PASS"
        if complete and all(validity[name] for name in required)
        else ("INVALID" if complete else "FAIL")
    )
    median_ms = statistics.median(times) if times else math.nan
    global_batch = int(env.get("global_batch_samples", "0"))
    sequence = int(env.get("sequence_length_tokens", "0"))
    metrics = {"steady_iterations": len(times)}
    if math.isfinite(median_ms):
        metrics["steady_iteration_time_ms"] = median_ms
        if median_ms > 0:
            metrics["tokens_per_second"] = global_batch * sequence * 1000.0 / median_ms
    if tflops and finite(tflops):
        metrics["tflops_per_gpu"] = statistics.mean(tflops)
    if losses and math.isfinite(losses[-1]):
        metrics["loss_last"] = losses[-1]
    if gradients and math.isfinite(gradients[-1]):
        metrics["gradient_norm_last"] = gradients[-1]
    cell = env.get("cell", run.parent.parent.name)
    repeat_text = env.get("repeat", run.parent.name.removeprefix("repeat-"))
    result = {
        "schema_version": 1,
        "campaign_id": env.get("campaign_id", "unknown"),
        "model": "kimi-k2",
        "cell": cell,
        "repeat": int(repeat_text),
        "ep_arm": arm,
        "status": status,
        "metrics": metrics,
        "validity": validity,
        "required_validity_gates": required,
        "artifacts": {
            "run_directory": str(run),
            "pod_logs": len(log_paths),
            "runtime_manifests": len(manifests),
            "route_summaries": len(routes),
            **iteration_artifacts,
        },
    }
    (run / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def bootstrap_ci(values: list[float], samples: int = 10000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(1234)
    draws = []
    for _ in range(samples):
        draws.append(statistics.mean(rng.choice(values) for _ in values))
    draws.sort()
    return [draws[int(0.025 * samples)], draws[int(0.975 * samples)]]


def summarize(results: list[dict]) -> dict:
    groups: dict[tuple[str, str], list[dict]] = {}
    for result in results:
        if result["status"] == "PASS":
            groups.setdefault((result["cell"], result["ep_arm"]), []).append(result)
    cells: dict[str, dict] = {}
    for (cell, arm), items in sorted(groups.items()):
        values = [item["metrics"]["steady_iteration_time_ms"] for item in items]
        mean = statistics.mean(values)
        cv = statistics.stdev(values) / mean if len(values) > 1 and mean else 0.0
        cells.setdefault(cell, {})[arm] = {
            "independent_job_starts": len(values),
            "median_of_run_medians_ms": statistics.median(values),
            "run_to_run_cv_dimensionless": cv,
        }
    paired: dict[str, dict] = {}
    for cell in cells:
        baseline = {
            item["repeat"]: item for item in groups.get((cell, "nccl-alltoall"), [])
        }
        for arm in ("uccl", "deepep-v1-nvshmem", "deepep-v2-gin-gda"):
            treatment = {item["repeat"]: item for item in groups.get((cell, arm), [])}
            deltas = []
            for repeat in sorted(set(baseline) & set(treatment)):
                base = baseline[repeat]["metrics"]["steady_iteration_time_ms"]
                value = treatment[repeat]["metrics"]["steady_iteration_time_ms"]
                deltas.append((base - value) / base * 100.0)
            paired.setdefault(cell, {})[arm] = {
                "paired_speedup_percent": statistics.mean(deltas) if deltas else None,
                "bootstrap_95_percent_ci": bootstrap_ci(deltas),
                "paired_repeats": len(deltas),
            }
    return {"cells": cells, "paired_vs_nccl_alltoall": paired}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.campaign)
    runs = sorted(path for path in root.glob("*/repeat-*/*") if path.is_dir())
    results = [parse_run(path, args.warmup) for path in runs]
    document = {
        "schema_version": 1,
        "campaign_root": str(root),
        "campaign_tree_sha256": hashlib.sha256(
            "\n".join(str(path) for path in runs).encode()
        ).hexdigest(),
        "runs": results,
        "summary": summarize(results),
    }
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
