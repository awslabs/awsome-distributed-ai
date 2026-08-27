#!/usr/bin/env python3
"""Compare Kimi-K2 training curves across EP arms without mutating raw artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "kimi-k2-ep-training-output-v1"
from matplotlib import pyplot as plt  # noqa: E402

ARMS = (
    "nccl-alltoall",
    "uccl",
    "deepep-v1-nvshmem",
    "deepep-v2-gin-gda",
)
ARM_LABELS = {
    "nccl-alltoall": "NCCL all-to-all",
    "uccl": "UCCL",
    "deepep-v1-nvshmem": "DeepEP v1",
    "deepep-v2-gin-gda": "DeepEP v2",
}
ARM_COLORS = {
    "nccl-alltoall": "#4c78a8",
    "uccl": "#f58518",
    "deepep-v1-nvshmem": "#54a24b",
    "deepep-v2-gin-gda": "#e45756",
}
ARM_LINESTYLES = {
    "nccl-alltoall": "-",
    "uccl": "--",
    "deepep-v1-nvshmem": "-.",
    "deepep-v2-gin-gda": ":",
}
ARM_MARKERS = {
    "nccl-alltoall": "o",
    "uccl": "s",
    "deepep-v1-nvshmem": "^",
    "deepep-v2-gin-gda": "D",
}

ITERATION = re.compile(r"\biteration\s+(\d+)\s*/\s*(\d+)", re.I)
FIELDS = {
    "elapsed_time_ms": re.compile(
        r"elapsed time per iteration \(ms\):\s*([0-9.eE+-]+)", re.I
    ),
    "lm_loss_dimensionless": re.compile(r"\blm loss:\s*([0-9.eE+-]+)", re.I),
    "gradient_norm_parameter_gradient_units": re.compile(
        r"grad(?:ient)? norm:\s*([0-9.eE+-]+)", re.I
    ),
    "learning_rate_dimensionless": re.compile(
        r"\blearning rate:\s*([0-9.eE+-]+)", re.I
    ),
    "skipped_iterations_count": re.compile(
        r"\bnumber of skipped iterations:\s*(\d+)", re.I
    ),
    "nan_iterations_count": re.compile(r"\bnumber of nan iterations:\s*(\d+)", re.I),
}
INTEGER_FIELDS = {"skipped_iterations_count", "nan_iterations_count"}
UPDATE_PREFIX = "OPTIMIZER_UPDATE_SAMPLE "
CONTROL_FIELDS = (
    "nodes",
    "world_size",
    "tp",
    "pp",
    "ep",
    "etp",
    "train_iterations",
    "global_batch_samples",
    "micro_batch_samples",
    "sequence_length_tokens",
    "ep_overlap",
    "performance_seed",
    "benchmark_learning_rate",
    "run_kind",
    "benchmark_entrypoint_source_sha256",
)


@dataclass(frozen=True)
class Run:
    path: Path
    environment: dict[str, str]
    records: tuple[dict[str, float | int], ...]
    route_hashes: tuple[str, ...]
    update_samples: tuple[dict, ...]

    @property
    def arm(self) -> str:
        return self.environment.get("ep_arm", self.path.name)

    @property
    def cell(self) -> str:
        return self.environment.get("cell", self.path.parent.parent.name)

    @property
    def repeat(self) -> int:
        return int(
            self.environment.get(
                "repeat", self.path.parent.name.removeprefix("repeat-")
            )
        )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_environment(run_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in (run_path / "environment.txt").read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def parse_records(text: str) -> tuple[dict[str, float | int], ...]:
    records: dict[int, dict[str, float | int]] = {}
    for line in text.splitlines():
        match = ITERATION.search(line)
        if match is None:
            continue
        iteration = int(match.group(1))
        record = records.setdefault(
            iteration,
            {
                "iteration_dimensionless": iteration,
                "total_iterations_dimensionless": int(match.group(2)),
            },
        )
        for name, pattern in FIELDS.items():
            value = pattern.search(line)
            if value is not None:
                record[name] = (
                    int(value.group(1))
                    if name in INTEGER_FIELDS
                    else float(value.group(1))
                )
    return tuple(records[index] for index in sorted(records))


def read_update_samples(run_path: Path) -> tuple[dict, ...]:
    candidates = (
        run_path / "node-0" / "node-rank-0.log",
        run_path / "pod-logs" / "node-rank-0.log",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return ()
    samples: dict[int, dict] = {}
    for line in path.read_text(errors="replace").splitlines():
        if UPDATE_PREFIX not in line:
            continue
        try:
            payload = json.loads(line.partition(UPDATE_PREFIX)[2])
        except json.JSONDecodeError:
            continue
        samples[int(payload["iteration"])] = payload
    return tuple(samples[index] for index in sorted(samples))


def read_route_hashes(run_path: Path) -> tuple[str, ...]:
    paths = sorted(
        run_path.glob("node-*/route-summary.json"),
        key=lambda path: int(path.parent.name.removeprefix("node-")),
    )
    hashes = []
    for path in paths:
        payload = json.loads(path.read_text())
        value = payload.get("topk_index_sha256")
        if value:
            hashes.append(str(value))
    return tuple(hashes)


def load_run(run_path: Path) -> Run:
    log_paths = sorted(run_path.glob("pod-logs/node-rank-*.log"))
    if not log_paths:
        raise ValueError(f"no trainer logs found in {run_path}")
    text = "\n".join(path.read_text(errors="replace") for path in log_paths)
    return Run(
        path=run_path,
        environment=read_environment(run_path),
        records=parse_records(text),
        route_hashes=read_route_hashes(run_path),
        update_samples=read_update_samples(run_path),
    )


def discover_runs(inputs: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for path in inputs:
        if (path / "environment.txt").is_file():
            found.add(path.resolve())
            continue
        for environment in path.glob("**/environment.txt"):
            candidate = environment.parent
            if candidate.name in ARMS:
                found.add(candidate.resolve())
    return sorted(found)


def bf16_tolerances(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    raw = payload.get("tolerance", payload)
    names = (
        "loss",
        "d_expert_parameter",
        "d_input",
        "d_router_probability",
        "optimizer_step",
        "output",
    )
    result = {}
    for name in names:
        if name not in raw:
            continue
        value = float(raw[name])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid {name} tolerance in {path}: {value}")
        result[name] = value
    if "loss" not in result:
        raise ValueError(f"loss tolerance is missing from {path}")
    return result


def norm_tolerances(
    tolerances: dict[str, float], sampled_elements: int
) -> tuple[float | None, float | None]:
    """Lift preserved elementwise BF16 bounds to conservative L2 bounds."""
    if sampled_elements <= 0:
        return None, None
    gradient_components = [
        tolerances[name]
        for name in ("d_expert_parameter", "d_input", "d_router_probability")
        if name in tolerances
    ]
    gradient = (
        max(gradient_components) * math.sqrt(sampled_elements)
        if gradient_components
        else None
    )
    update = (
        tolerances["optimizer_step"] * math.sqrt(sampled_elements)
        if "optimizer_step" in tolerances
        else None
    )
    return gradient, update


def finite_record(record: dict[str, float | int]) -> bool:
    return all(
        math.isfinite(float(record[name]))
        for name in (
            "elapsed_time_ms",
            "lm_loss_dimensionless",
            "gradient_norm_parameter_gradient_units",
            "learning_rate_dimensionless",
        )
        if name in record
    )


def compare_group(
    group: list[Run], tolerances: dict[str, float], require_route_hashes: bool
) -> dict:
    loss_tolerance = tolerances["loss"]
    by_arm = {run.arm: run for run in group}
    missing_arms = sorted(set(ARMS) - set(by_arm))
    duplicate_arms = sorted(
        arm for arm in ARMS if sum(run.arm == arm for run in group) != 1
    )
    baseline = by_arm.get("nccl-alltoall")
    common_controls: dict[str, dict[str, str]] = {}
    for field in CONTROL_FIELDS:
        values = {run.arm: run.environment.get(field, "") for run in group}
        if len(set(values.values())) != 1:
            common_controls[field] = values

    run_results = {}
    for run in group:
        expected = int(run.environment.get("train_iterations", "0"))
        iterations = [int(record["iteration_dimensionless"]) for record in run.records]
        complete = iterations == list(range(1, expected + 1))
        required_fields = all(
            {
                "lm_loss_dimensionless",
                "gradient_norm_parameter_gradient_units",
                "learning_rate_dimensionless",
                "skipped_iterations_count",
                "nan_iterations_count",
            }
            <= record.keys()
            for record in run.records
        )
        finite = bool(run.records) and all(
            finite_record(record) for record in run.records
        )
        no_skips = bool(run.records) and all(
            int(record.get("skipped_iterations_count", -1)) == 0
            and int(record.get("nan_iterations_count", -1)) == 0
            for record in run.records
        )
        lr_expected = float(run.environment.get("benchmark_learning_rate", "nan"))
        learning_rate_matches = math.isfinite(lr_expected) and all(
            math.isclose(
                float(record.get("learning_rate_dimensionless", math.nan)),
                lr_expected,
                rel_tol=0.0,
                abs_tol=max(abs(lr_expected) * 1e-7, 1e-15),
            )
            for record in run.records
        )
        update_samples_required = run.environment.get("run_kind") == "correctness"
        update_samples_valid = not update_samples_required or (
            len(run.update_samples) == expected
            and all(
                sample.get("finite") is True
                and sample.get("skipped_iteration") is False
                and all(
                    math.isfinite(float(sample[name]))
                    for name in (
                        "gradient_norm",
                        "update_l2_parameter_units",
                        "update_max_abs_parameter_units",
                        "update_mean_abs_parameter_units",
                    )
                )
                for sample in run.update_samples
            )
        )
        expected_nodes = len(
            [node for node in run.environment.get("nodes", "").split(",") if node]
        )
        route_hashes_complete = (
            expected_nodes > 0 and len(run.route_hashes) == expected_nodes
        )
        run_results[run.arm] = {
            "artifact_path": str(run.path),
            "environment_sha256": sha256(run.path / "environment.txt"),
            "iterations_observed_dimensionless": len(run.records),
            "iterations_expected_dimensionless": expected,
            "iteration_metrics_complete": complete and required_fields,
            "all_metrics_finite": finite,
            "no_skipped_or_nan_iterations": no_skips,
            "learning_rate_matches_environment": learning_rate_matches,
            "optimizer_update_samples_valid": update_samples_valid,
            "route_hashes_observed": len(run.route_hashes),
            "route_hashes_expected": expected_nodes,
            "route_hashes_complete": route_hashes_complete,
            "optimizer_update_samples_observed": len(run.update_samples),
            "loss_first_dimensionless": (
                float(run.records[0]["lm_loss_dimensionless"]) if run.records else None
            ),
            "loss_last_dimensionless": (
                float(run.records[-1]["lm_loss_dimensionless"]) if run.records else None
            ),
            "loss_min_dimensionless": (
                min(float(record["lm_loss_dimensionless"]) for record in run.records)
                if run.records
                else None
            ),
            "loss_max_dimensionless": (
                max(float(record["lm_loss_dimensionless"]) for record in run.records)
                if run.records
                else None
            ),
        }

    comparisons = {}
    if baseline is not None:
        baseline_by_iteration = {
            int(record["iteration_dimensionless"]): record
            for record in baseline.records
        }
        baseline_updates = {
            int(sample["iteration"]): sample for sample in baseline.update_samples
        }
        for arm, run in by_arm.items():
            if arm == "nccl-alltoall":
                continue
            deltas = []
            logged_gradient_deltas = []
            for record in run.records:
                iteration = int(record["iteration_dimensionless"])
                reference = baseline_by_iteration.get(iteration)
                if reference is None:
                    continue
                delta = abs(
                    float(record["lm_loss_dimensionless"])
                    - float(reference["lm_loss_dimensionless"])
                )
                deltas.append(
                    {
                        "iteration_dimensionless": iteration,
                        "absolute_loss_delta_dimensionless": delta,
                        "within_nccl_self_repeat_tolerance": delta <= loss_tolerance,
                    }
                )
                logged_gradient_deltas.append(
                    abs(
                        float(record["gradient_norm_parameter_gradient_units"])
                        - float(reference["gradient_norm_parameter_gradient_units"])
                    )
                )
            run_updates = {
                int(sample["iteration"]): sample for sample in run.update_samples
            }
            update_iterations_match = bool(baseline_updates) and (
                set(run_updates) == set(baseline_updates)
            )
            update_layout_matches = update_iterations_match
            sampled_element_counts = set()
            precise_gradient_deltas = []
            update_deltas = []
            for iteration, sample in sorted(run_updates.items()):
                reference = baseline_updates.get(iteration)
                if reference is not None:
                    precise_gradient_deltas.append(
                        abs(
                            float(sample["gradient_norm"])
                            - float(reference["gradient_norm"])
                        )
                    )
                    update_deltas.append(
                        abs(
                            float(sample["update_l2_parameter_units"])
                            - float(reference["update_l2_parameter_units"])
                        )
                    )
                    for field in (
                        "aggregation",
                        "layout_sha256_rank_0",
                        "sample_limit_elements_per_rank",
                        "sampled_elements",
                    ):
                        if sample.get(field) != reference.get(field):
                            update_layout_matches = False
                    sampled_element_counts.update(
                        (
                            int(sample["sampled_elements"]),
                            int(reference["sampled_elements"]),
                        )
                    )
            gradient_deltas = precise_gradient_deltas or logged_gradient_deltas
            sampled_elements = (
                sampled_element_counts.pop() if len(sampled_element_counts) == 1 else 0
            )
            gradient_tolerance, update_tolerance = norm_tolerances(
                tolerances, sampled_elements
            )
            short_output_gate_required = (
                baseline.environment.get("run_kind") == "correctness"
            )
            gradient_gate = (
                bool(precise_gradient_deltas)
                and gradient_tolerance is not None
                and max(precise_gradient_deltas) <= gradient_tolerance
            )
            update_gate = (
                bool(update_deltas)
                and update_tolerance is not None
                and max(update_deltas) <= update_tolerance
            )
            output_norm_gate = (
                update_layout_matches and gradient_gate and update_gate
                if short_output_gate_required
                else True
            )
            route_hashes_match = (
                bool(baseline.route_hashes)
                and len(run.route_hashes) == len(baseline.route_hashes)
                and run.route_hashes == baseline.route_hashes
            )
            comparisons[arm] = {
                "iterations_compared_dimensionless": len(deltas),
                "max_absolute_loss_delta_dimensionless": (
                    max(item["absolute_loss_delta_dimensionless"] for item in deltas)
                    if deltas
                    else None
                ),
                "max_absolute_gradient_norm_delta_parameter_gradient_units": (
                    max(gradient_deltas) if gradient_deltas else None
                ),
                "max_absolute_sampled_update_l2_delta_parameter_units": (
                    max(update_deltas) if update_deltas else None
                ),
                "gradient_norm_tolerance_parameter_gradient_units": gradient_tolerance,
                "sampled_update_l2_tolerance_parameter_units": update_tolerance,
                "sampled_elements_dimensionless": sampled_elements,
                "gradient_norm_comparison_source": (
                    "optimizer_update_samples_full_precision"
                    if precise_gradient_deltas
                    else "iteration_log_rounded"
                ),
                "gradient_norm_within_nccl_bf16_bound": gradient_gate,
                "sampled_update_l2_within_nccl_bf16_bound": update_gate,
                "optimizer_update_sample_layout_matches_nccl": update_layout_matches,
                "short_training_output_norm_gate_required": short_output_gate_required,
                "short_training_output_norm_gate": output_norm_gate,
                "loss_curve_within_nccl_self_repeat_tolerance": bool(deltas)
                and all(item["within_nccl_self_repeat_tolerance"] for item in deltas),
                "first_failing_iteration_dimensionless": next(
                    (
                        item["iteration_dimensionless"]
                        for item in deltas
                        if not item["within_nccl_self_repeat_tolerance"]
                    ),
                    None,
                ),
                "route_hashes_match_nccl": route_hashes_match,
                "per_iteration": deltas,
            }

    individual_pass = all(
        result["iteration_metrics_complete"]
        and result["all_metrics_finite"]
        and result["no_skipped_or_nan_iterations"]
        and result["learning_rate_matches_environment"]
        and result["optimizer_update_samples_valid"]
        and (result["route_hashes_complete"] or not require_route_hashes)
        for result in run_results.values()
    )
    comparison_pass = bool(comparisons) and all(
        item["loss_curve_within_nccl_self_repeat_tolerance"]
        and item["short_training_output_norm_gate"]
        and (item["route_hashes_match_nccl"] or not require_route_hashes)
        for item in comparisons.values()
    )
    passed = (
        not missing_arms
        and not duplicate_arms
        and not common_controls
        and individual_pass
        and comparison_pass
    )
    return {
        "cell": group[0].cell,
        "repeat_dimensionless": group[0].repeat,
        "status": "PASS" if passed else "FAIL",
        "missing_arms": missing_arms,
        "duplicate_or_missing_arm_cardinality": duplicate_arms,
        "common_control_mismatches": common_controls,
        "runs": run_results,
        "comparisons_vs_nccl_alltoall": comparisons,
    }


def compare_nccl_training_repeat(
    baseline: Run, repeat: Run, tolerances: dict[str, float]
) -> dict:
    if baseline.arm != "nccl-alltoall" or repeat.arm != "nccl-alltoall":
        raise ValueError(
            "NCCL training repeat comparison requires two nccl-alltoall runs"
        )
    control_mismatches = {}
    for field in CONTROL_FIELDS:
        left = baseline.environment.get(field, "")
        right = repeat.environment.get(field, "")
        if left != right:
            control_mismatches[field] = {"baseline": left, "repeat": right}

    baseline_records = {
        int(record["iteration_dimensionless"]): record for record in baseline.records
    }
    repeat_records = {
        int(record["iteration_dimensionless"]): record for record in repeat.records
    }
    record_iterations_match = bool(baseline_records) and (
        set(baseline_records) == set(repeat_records)
    )
    loss_deltas = []
    if record_iterations_match:
        for iteration in sorted(baseline_records):
            loss_deltas.append(
                {
                    "iteration_dimensionless": iteration,
                    "absolute_loss_delta_dimensionless": abs(
                        float(baseline_records[iteration]["lm_loss_dimensionless"])
                        - float(repeat_records[iteration]["lm_loss_dimensionless"])
                    ),
                }
            )

    baseline_updates = {
        int(sample["iteration"]): sample for sample in baseline.update_samples
    }
    repeat_updates = {
        int(sample["iteration"]): sample for sample in repeat.update_samples
    }
    update_iterations_match = bool(baseline_updates) and (
        set(baseline_updates) == set(repeat_updates)
    )
    update_layout_matches = update_iterations_match
    sampled_element_counts = set()
    gradient_deltas = []
    update_deltas = []
    if update_iterations_match:
        for iteration in sorted(baseline_updates):
            left = baseline_updates[iteration]
            right = repeat_updates[iteration]
            gradient_deltas.append(
                abs(float(left["gradient_norm"]) - float(right["gradient_norm"]))
            )
            update_deltas.append(
                abs(
                    float(left["update_l2_parameter_units"])
                    - float(right["update_l2_parameter_units"])
                )
            )
            for field in (
                "aggregation",
                "layout_sha256_rank_0",
                "sample_limit_elements_per_rank",
                "sampled_elements",
            ):
                if left.get(field) != right.get(field):
                    update_layout_matches = False
            sampled_element_counts.update(
                (int(left["sampled_elements"]), int(right["sampled_elements"]))
            )
    sampled_elements = (
        sampled_element_counts.pop() if len(sampled_element_counts) == 1 else 0
    )
    gradient_tolerance, update_tolerance = norm_tolerances(tolerances, sampled_elements)
    max_loss_delta = (
        max(item["absolute_loss_delta_dimensionless"] for item in loss_deltas)
        if loss_deltas
        else None
    )
    max_gradient_delta = max(gradient_deltas) if gradient_deltas else None
    max_update_delta = max(update_deltas) if update_deltas else None
    route_hashes_match = (
        bool(baseline.route_hashes)
        and baseline.route_hashes == repeat.route_hashes
        and len(baseline.route_hashes)
        == len(
            [node for node in baseline.environment.get("nodes", "").split(",") if node]
        )
    )
    baseline_status_pass = (baseline.path / "STATUS").read_text().startswith("PASS")
    repeat_status_pass = (repeat.path / "STATUS").read_text().startswith("PASS")
    no_skipped_or_nan = all(
        int(record.get("skipped_iterations_count", -1)) == 0
        and int(record.get("nan_iterations_count", -1)) == 0
        for record in (*baseline.records, *repeat.records)
    )
    finite_updates = all(
        sample.get("finite") is True and sample.get("skipped_iteration") is False
        for sample in (*baseline.update_samples, *repeat.update_samples)
    )
    loss_gate = max_loss_delta is not None and max_loss_delta <= tolerances["loss"]
    gradient_gate = (
        max_gradient_delta is not None
        and gradient_tolerance is not None
        and max_gradient_delta <= gradient_tolerance
    )
    update_gate = (
        max_update_delta is not None
        and update_tolerance is not None
        and max_update_delta <= update_tolerance
    )
    passed = all(
        (
            baseline_status_pass,
            repeat_status_pass,
            not control_mismatches,
            record_iterations_match,
            update_iterations_match,
            update_layout_matches,
            route_hashes_match,
            no_skipped_or_nan,
            finite_updates,
            loss_gate,
            gradient_gate,
            update_gate,
        )
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "baseline_artifact_path": str(baseline.path),
        "repeat_artifact_path": str(repeat.path),
        "baseline_repeat_dimensionless": baseline.repeat,
        "repeat_dimensionless": repeat.repeat,
        "control_mismatches": control_mismatches,
        "artifact_status_pass": baseline_status_pass and repeat_status_pass,
        "iteration_indices_match": record_iterations_match,
        "optimizer_update_iteration_indices_match": update_iterations_match,
        "optimizer_update_sample_layout_matches": update_layout_matches,
        "route_hashes_match": route_hashes_match,
        "route_hashes_observed": len(repeat.route_hashes),
        "no_skipped_or_nan_iterations": no_skipped_or_nan,
        "optimizer_update_samples_finite": finite_updates,
        "sampled_elements_dimensionless": sampled_elements,
        "max_absolute_loss_delta_dimensionless": max_loss_delta,
        "loss_tolerance_dimensionless": tolerances["loss"],
        "loss_gate": loss_gate,
        "max_absolute_gradient_norm_delta_parameter_gradient_units": max_gradient_delta,
        "gradient_norm_tolerance_parameter_gradient_units": gradient_tolerance,
        "gradient_norm_gate": gradient_gate,
        "max_absolute_sampled_update_l2_delta_parameter_units": max_update_delta,
        "sampled_update_l2_tolerance_parameter_units": update_tolerance,
        "sampled_update_l2_gate": update_gate,
        "per_iteration_loss_deltas": loss_deltas,
    }


def write_combined_csv(runs: list[Run], output_path: Path) -> None:
    fields = [
        "cell",
        "repeat_dimensionless",
        "ep_arm",
        "iteration_dimensionless",
        "lm_loss_dimensionless",
        "gradient_norm_parameter_gradient_units",
        "learning_rate_dimensionless",
        "elapsed_time_ms",
        "skipped_iterations_count",
        "nan_iterations_count",
    ]
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in sorted(
            runs, key=lambda item: (item.cell, item.repeat, ARMS.index(item.arm))
        ):
            for record in run.records:
                writer.writerow(
                    {
                        "cell": run.cell,
                        "repeat_dimensionless": run.repeat,
                        "ep_arm": run.arm,
                        **{field: record.get(field, "") for field in fields[3:]},
                    }
                )


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def format_optional_metric(value: float | int | None) -> str:
    return "Not recorded" if value is None else f"{value:.10g}"


def plot_cell(cell: str, runs: list[Run], output_dir: Path) -> list[str]:
    repeats = sorted({run.repeat for run in runs})
    figure, axes = plt.subplots(
        len(repeats),
        1,
        figsize=(9.2, 3.7 * len(repeats)),
        sharex=True,
        squeeze=False,
        constrained_layout=False,
    )
    for axis, repeat in zip(axes[:, 0], repeats, strict=True):
        selected = [run for run in runs if run.repeat == repeat]
        all_losses = []
        for run in sorted(selected, key=lambda item: ARMS.index(item.arm)):
            iterations = [int(item["iteration_dimensionless"]) for item in run.records]
            losses = [float(item["lm_loss_dimensionless"]) for item in run.records]
            all_losses.extend(losses)
            axis.plot(
                iterations,
                losses,
                label=ARM_LABELS[run.arm],
                color=ARM_COLORS[run.arm],
                linewidth=1.9,
                linestyle=ARM_LINESTYLES[run.arm],
                marker=ARM_MARKERS[run.arm] if len(iterations) <= 8 else None,
                markersize=3.5,
            )
        if (
            all_losses
            and min(all_losses) > 0.0
            and max(all_losses) / min(all_losses) >= 10.0
        ):
            axis.set_yscale("log")
        axis.text(
            0.01,
            0.97,
            f"Repeat {repeat} (dimensionless)",
            transform=axis.transAxes,
            ha="left",
            va="top",
        )
        axis.set_ylabel("LM loss (dimensionless)")
        axis.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("Optimizer iteration (dimensionless)")
    figure.subplots_adjust(top=0.78, bottom=0.16, left=0.1, right=0.98, hspace=0.4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.9),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(f"Kimi-K2 EP arm loss curves: {cell}", fontsize=14, y=0.98)
    stem = f"loss-curves-{safe_name(cell)}"
    paths = []
    for suffix in ("svg", "png"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(
            path, dpi=180, metadata={"Date": None} if suffix == "svg" else None
        )
        paths.append(path.name)
    plt.close(figure)
    return paths


def plot_training_outputs(cell: str, runs: list[Run], output_dir: Path) -> list[str]:
    repeats = sorted({run.repeat for run in runs})
    has_update_samples = any(run.update_samples for run in runs)
    column_count = 3 if has_update_samples else 2
    figure, axes = plt.subplots(
        len(repeats),
        column_count,
        figsize=((15.2 if has_update_samples else 13.2), 3.7 * len(repeats)),
        squeeze=False,
        constrained_layout=False,
    )
    for row, repeat in enumerate(repeats):
        selected = [run for run in runs if run.repeat == repeat]
        for run in sorted(selected, key=lambda item: ARMS.index(item.arm)):
            record_iterations = [
                int(item["iteration_dimensionless"]) for item in run.records
            ]
            losses = [float(item["lm_loss_dimensionless"]) for item in run.records]
            axes[row, 0].plot(
                record_iterations,
                losses,
                label=ARM_LABELS[run.arm],
                color=ARM_COLORS[run.arm],
                linewidth=1.9,
                linestyle=ARM_LINESTYLES[run.arm],
                marker=ARM_MARKERS[run.arm],
                markersize=3.5,
            )
            if run.update_samples:
                gradient_iterations = [
                    int(sample["iteration"]) + 1 for sample in run.update_samples
                ]
                gradients = [
                    float(sample["gradient_norm"]) for sample in run.update_samples
                ]
            else:
                gradient_iterations = record_iterations
                gradients = [
                    float(item["gradient_norm_parameter_gradient_units"])
                    for item in run.records
                ]
            axes[row, 1].plot(
                gradient_iterations,
                gradients,
                color=ARM_COLORS[run.arm],
                linewidth=1.9,
                linestyle=ARM_LINESTYLES[run.arm],
                marker=ARM_MARKERS[run.arm],
                markersize=3.5,
            )
            if run.update_samples:
                update_iterations = [
                    int(sample["iteration"]) + 1 for sample in run.update_samples
                ]
                updates = [
                    float(sample["update_l2_parameter_units"])
                    for sample in run.update_samples
                ]
                axes[row, 2].plot(
                    update_iterations,
                    updates,
                    color=ARM_COLORS[run.arm],
                    linewidth=1.9,
                    linestyle=ARM_LINESTYLES[run.arm],
                    marker=ARM_MARKERS[run.arm],
                    markersize=3.5,
                )
        axes[row, 0].set_ylabel(
            f"Repeat {repeat} (dimensionless)\nLM loss (dimensionless)"
        )
        axes[row, 1].set_ylabel("Gradient norm\n(parameter-gradient units)")
        if has_update_samples:
            axes[row, 2].set_ylabel("Sampled update L2\n(parameter units)")
        for axis in axes[row, :]:
            axis.grid(True, alpha=0.25)
            axis.set_xlabel("Optimizer iteration (dimensionless)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.subplots_adjust(
        top=0.78, bottom=0.16, left=0.08, right=0.98, hspace=0.45, wspace=0.3
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.9),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(f"Kimi-K2 EP arm training outputs: {cell}", fontsize=14, y=0.98)
    stem = f"training-output-curves-{safe_name(cell)}"
    paths = []
    for suffix in ("svg", "png"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(
            path, dpi=180, metadata={"Date": None} if suffix == "svg" else None
        )
        paths.append(path.name)
    plt.close(figure)
    return paths


def plot_output_deltas(
    cell: str, runs: list[Run], tolerances: dict[str, float], output_dir: Path
) -> list[str]:
    repeats = sorted({run.repeat for run in runs})
    has_update_samples = any(run.update_samples for run in runs)
    column_count = 3 if has_update_samples else 2
    figure, axes = plt.subplots(
        len(repeats),
        column_count,
        figsize=((15.2 if has_update_samples else 13.2), 3.7 * len(repeats)),
        squeeze=False,
        constrained_layout=False,
    )
    for row, repeat in enumerate(repeats):
        selected = [run for run in runs if run.repeat == repeat]
        baseline = next((run for run in selected if run.arm == "nccl-alltoall"), None)
        if baseline is None:
            continue
        baseline_records = {
            int(record["iteration_dimensionless"]): record
            for record in baseline.records
        }
        baseline_updates = {
            int(sample["iteration"]): sample for sample in baseline.update_samples
        }
        sampled_elements = (
            int(baseline.update_samples[0]["sampled_elements"])
            if baseline.update_samples
            else 0
        )
        gradient_tolerance, update_tolerance = norm_tolerances(
            tolerances, sampled_elements
        )
        for run in sorted(selected, key=lambda item: ARMS.index(item.arm)):
            if run.arm == "nccl-alltoall":
                continue
            iterations = [
                int(record["iteration_dimensionless"]) for record in run.records
            ]
            loss_deltas = [
                abs(
                    float(record["lm_loss_dimensionless"])
                    - float(
                        baseline_records[int(record["iteration_dimensionless"])][
                            "lm_loss_dimensionless"
                        ]
                    )
                )
                for record in run.records
            ]
            axes[row, 0].plot(
                iterations,
                loss_deltas,
                label=ARM_LABELS[run.arm],
                color=ARM_COLORS[run.arm],
                linestyle=ARM_LINESTYLES[run.arm],
                marker=ARM_MARKERS[run.arm],
                linewidth=1.9,
                markersize=3.5,
            )
            if run.update_samples and baseline_updates:
                gradient_iterations = [
                    int(sample["iteration"]) + 1 for sample in run.update_samples
                ]
                gradient_deltas = [
                    abs(
                        float(sample["gradient_norm"])
                        - float(
                            baseline_updates[int(sample["iteration"])]["gradient_norm"]
                        )
                    )
                    for sample in run.update_samples
                ]
            else:
                gradient_iterations = iterations
                gradient_deltas = [
                    abs(
                        float(record["gradient_norm_parameter_gradient_units"])
                        - float(
                            baseline_records[int(record["iteration_dimensionless"])][
                                "gradient_norm_parameter_gradient_units"
                            ]
                        )
                    )
                    for record in run.records
                ]
            axes[row, 1].plot(
                gradient_iterations,
                gradient_deltas,
                color=ARM_COLORS[run.arm],
                linestyle=ARM_LINESTYLES[run.arm],
                marker=ARM_MARKERS[run.arm],
                linewidth=1.9,
                markersize=3.5,
            )
            if run.update_samples and baseline_updates:
                update_iterations = [
                    int(sample["iteration"]) + 1 for sample in run.update_samples
                ]
                update_deltas = [
                    abs(
                        float(sample["update_l2_parameter_units"])
                        - float(
                            baseline_updates[int(sample["iteration"])][
                                "update_l2_parameter_units"
                            ]
                        )
                    )
                    for sample in run.update_samples
                ]
                axes[row, 2].plot(
                    update_iterations,
                    update_deltas,
                    color=ARM_COLORS[run.arm],
                    linestyle=ARM_LINESTYLES[run.arm],
                    marker=ARM_MARKERS[run.arm],
                    linewidth=1.9,
                    markersize=3.5,
                )
        axes[row, 0].axhline(
            tolerances["loss"], color="#666666", linestyle=":", label="NCCL BF16 bound"
        )
        if gradient_tolerance is not None:
            axes[row, 1].axhline(gradient_tolerance, color="#666666", linestyle=":")
        if has_update_samples and update_tolerance is not None:
            axes[row, 2].axhline(update_tolerance, color="#666666", linestyle=":")
        axes[row, 0].text(
            0.01,
            0.97,
            f"Repeat {repeat} (dimensionless)",
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
        )
        axes[row, 0].set_ylabel("Absolute loss delta\n(dimensionless)")
        gradient_label = (
            "Absolute gradient-norm delta\n(parameter-gradient units)"
            if has_update_samples
            else "Logged gradient-norm delta\n(parameter-gradient units; rounded)"
        )
        axes[row, 1].set_ylabel(gradient_label)
        if has_update_samples:
            axes[row, 2].set_ylabel("Absolute update-L2 delta\n(parameter units)")
        for axis in axes[row, :]:
            axis.grid(True, which="both", alpha=0.25)
            axis.set_xlabel("Optimizer iteration (dimensionless)")
        axes[row, 0].set_yscale("log")
        if has_update_samples:
            axes[row, 1].set_yscale("log")
            axes[row, 2].set_yscale("log")
        else:
            axes[row, 1].ticklabel_format(
                axis="y", style="sci", scilimits=(-3, -3), useMathText=True
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.subplots_adjust(
        top=0.78, bottom=0.16, left=0.09, right=0.98, hspace=0.45, wspace=0.38
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.9),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        f"Kimi-K2 EP arm output deltas vs NCCL: {cell}", fontsize=14, y=0.98
    )
    stem = f"training-output-deltas-{safe_name(cell)}"
    paths = []
    for suffix in ("svg", "png"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(
            path, dpi=180, metadata={"Date": None} if suffix == "svg" else None
        )
        paths.append(path.name)
    plt.close(figure)
    return paths


def write_markdown(document: dict, output_path: Path) -> None:
    lines = [
        "# Kimi-K2 EP arm training-output comparison",
        "",
        f"Overall status: `{document['status']}`.",
        "",
        (
            "The absolute loss tolerance is "
            f"{document['loss_tolerance_dimensionless']:.10g} dimensionless and comes from "
            "the preserved NCCL BF16 self-repeat envelope."
        ),
        (
            "Gradient-norm and sampled update-L2 bounds are derived without a fitted "
            "multiplier as `elementwise NCCL BF16 tolerance * sqrt(sampled elements)`."
        ),
        (
            "For performance runs without full-precision optimizer samples, the rounded "
            "gradient norms from iteration logs are plotted as diagnostics and are not "
            "used as a numeric gate."
        ),
        "",
    ]
    self_repeat = document.get("nccl_training_self_repeat")
    if self_repeat is not None:
        lines.extend(
            [
                "## NCCL training self-repeat",
                "",
                f"Self-repeat status: `{self_repeat['status']}`.",
                "",
                "| Metric | Maximum absolute delta | NCCL BF16-derived bound | Gate |",
                "|---|---:|---:|---|",
                (
                    "| LM loss, dimensionless | "
                    f"{self_repeat['max_absolute_loss_delta_dimensionless']:.10g} | "
                    f"{self_repeat['loss_tolerance_dimensionless']:.10g} | "
                    f"{'PASS' if self_repeat['loss_gate'] else 'FAIL'} |"
                ),
                (
                    "| Gradient norm, parameter-gradient units | "
                    f"{self_repeat['max_absolute_gradient_norm_delta_parameter_gradient_units']:.10g} | "
                    f"{self_repeat['gradient_norm_tolerance_parameter_gradient_units']:.10g} | "
                    f"{'PASS' if self_repeat['gradient_norm_gate'] else 'FAIL'} |"
                ),
                (
                    "| Sampled optimizer update L2, parameter units | "
                    f"{self_repeat['max_absolute_sampled_update_l2_delta_parameter_units']:.10g} | "
                    f"{self_repeat['sampled_update_l2_tolerance_parameter_units']:.10g} | "
                    f"{'PASS' if self_repeat['sampled_update_l2_gate'] else 'FAIL'} |"
                ),
                "",
            ]
        )
    last_repeat_by_cell = {
        cell: max(
            group["repeat_dimensionless"]
            for group in document["groups"]
            if group["cell"] == cell
        )
        for cell in {group["cell"] for group in document["groups"]}
    }
    for group in document["groups"]:
        lines.extend(
            [
                f"## {group['cell']}, repeat {group['repeat_dimensionless']} (dimensionless)",
                "",
                f"Group status: `{group['status']}`.",
                "",
                "| Arm | Iterations, dimensionless | First loss, dimensionless | Last loss, dimensionless | Maximum absolute loss delta vs NCCL, dimensionless | Maximum gradient-norm delta / bound, parameter-gradient units | Maximum sampled update-L2 delta / bound, parameter units | Loss gate | Gradient gate | Update gate | Route hash gate |",
                "|---|---:|---:|---:|---:|---:|---:|---|---|---|---|",
            ]
        )
        for arm in ARMS:
            run = group["runs"].get(arm)
            if run is None:
                continue
            comparison = group["comparisons_vs_nccl_alltoall"].get(arm)
            loss_delta_value = (
                None
                if arm == "nccl-alltoall"
                else comparison["max_absolute_loss_delta_dimensionless"]
            )
            max_delta = (
                "Reference"
                if arm == "nccl-alltoall"
                else "Not comparable"
                if loss_delta_value is None
                else f"{loss_delta_value:.10g}"
            )
            curve_gate = (
                "Reference"
                if arm == "nccl-alltoall"
                else (
                    "PASS"
                    if comparison["loss_curve_within_nccl_self_repeat_tolerance"]
                    else "FAIL"
                )
            )
            route_gate = (
                "Reference"
                if arm == "nccl-alltoall"
                else ("PASS" if comparison["route_hashes_match_nccl"] else "FAIL")
            )
            output_norm_gate_required = (
                False
                if arm == "nccl-alltoall"
                else comparison["short_training_output_norm_gate_required"]
            )
            gradient_value = (
                None
                if arm == "nccl-alltoall"
                else comparison[
                    "max_absolute_gradient_norm_delta_parameter_gradient_units"
                ]
            )
            gradient_bound = (
                None
                if arm == "nccl-alltoall"
                else comparison["gradient_norm_tolerance_parameter_gradient_units"]
            )
            gradient_delta = (
                "Reference"
                if arm == "nccl-alltoall"
                else "Not recorded"
                if gradient_value is None
                else f"{gradient_value:.10g} / Not gated"
                if not output_norm_gate_required or gradient_bound is None
                else f"{gradient_value:.10g} / {gradient_bound:.10g}"
            )
            update_value = (
                None
                if arm == "nccl-alltoall"
                else comparison["max_absolute_sampled_update_l2_delta_parameter_units"]
            )
            update_delta = (
                "Reference"
                if arm == "nccl-alltoall"
                else (
                    "Not recorded"
                    if update_value is None
                    else f"{update_value:.10g} / Not gated"
                    if not output_norm_gate_required
                    else (
                        f"{update_value:.10g} / "
                        f"{comparison['sampled_update_l2_tolerance_parameter_units']:.10g}"
                    )
                )
            )
            gradient_gate = (
                "Reference"
                if arm == "nccl-alltoall"
                else "Not required"
                if not output_norm_gate_required
                else (
                    "PASS"
                    if comparison["gradient_norm_within_nccl_bf16_bound"]
                    else "FAIL"
                )
            )
            update_gate = (
                "Reference"
                if arm == "nccl-alltoall"
                else "Not required"
                if not output_norm_gate_required
                else (
                    "PASS"
                    if comparison["sampled_update_l2_within_nccl_bf16_bound"]
                    else "FAIL"
                )
            )
            lines.append(
                f"| `{arm}` | {run['iterations_observed_dimensionless']} | "
                f"{format_optional_metric(run['loss_first_dimensionless'])} | "
                f"{format_optional_metric(run['loss_last_dimensionless'])} | "
                f"{max_delta} | {gradient_delta} | {update_delta} | {curve_gate} | "
                f"{gradient_gate} | {update_gate} | {route_gate} |"
            )
        lines.append("")
        if group["repeat_dimensionless"] != last_repeat_by_cell[group["cell"]]:
            continue
        figures = document["figures"].get(group["cell"], {})
        loss_figures = figures.get("loss_curves", [])
        output_figures = figures.get("training_output_curves", [])
        delta_figures = figures.get("training_output_deltas", [])
        if loss_figures:
            lines.extend(
                [
                    f"![Loss curves for {group['cell']}]({loss_figures[-1]})",
                    "",
                ]
            )
        if output_figures:
            lines.extend(
                [
                    f"![Training-output curves for {group['cell']}]({output_figures[-1]})",
                    "",
                ]
            )
        if delta_figures:
            lines.extend(
                [
                    f"![Training-output deltas for {group['cell']}]({delta_figures[-1]})",
                    "",
                ]
            )
    while lines and not lines[-1]:
        lines.pop()
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--loss-tolerance-json", required=True, type=Path)
    parser.add_argument("--nccl-self-repeat", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--require-route-hashes", action="store_true")
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()

    run_paths = discover_runs(args.inputs)
    if not run_paths:
        raise SystemExit("no run directories with environment.txt were found")
    runs = [load_run(path) for path in run_paths]
    unexpected = sorted({run.arm for run in runs} - set(ARMS))
    if unexpected:
        raise SystemExit(f"unexpected EP arms: {unexpected}")
    tolerances = bf16_tolerances(args.loss_tolerance_json)
    grouped: dict[tuple[str, int], list[Run]] = {}
    for run in runs:
        grouped.setdefault((run.cell, run.repeat), []).append(run)
    groups = [
        compare_group(group, tolerances, args.require_route_hashes)
        for _, group in sorted(grouped.items())
    ]
    nccl_training_self_repeat = None
    if args.nccl_self_repeat is not None:
        repeat = load_run(args.nccl_self_repeat.resolve())
        candidates = [
            run
            for run in runs
            if run.arm == "nccl-alltoall" and run.cell == repeat.cell
        ]
        if len(candidates) != 1:
            raise SystemExit(
                "exactly one matching NCCL baseline is required for --nccl-self-repeat"
            )
        nccl_training_self_repeat = compare_nccl_training_repeat(
            candidates[0], repeat, tolerances
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = args.output_dir / "iteration-metrics.csv"
    write_combined_csv(runs, combined_csv)
    figures = {}
    for cell in sorted({run.cell for run in runs}):
        cell_runs = [run for run in runs if run.cell == cell]
        figures[cell] = {
            "loss_curves": plot_cell(cell, cell_runs, args.output_dir),
            "training_output_curves": plot_training_outputs(
                cell, cell_runs, args.output_dir
            ),
            "training_output_deltas": plot_output_deltas(
                cell, cell_runs, tolerances, args.output_dir
            ),
        }
    document = {
        "schema_version": 1,
        "analysis_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "status": "PASS"
        if all(group["status"] == "PASS" for group in groups)
        and (
            nccl_training_self_repeat is None
            or nccl_training_self_repeat["status"] == "PASS"
        )
        else "FAIL",
        "loss_tolerance_dimensionless": tolerances["loss"],
        "nccl_bf16_elementwise_tolerances": tolerances,
        "norm_tolerance_derivation": "elementwise_tolerance * sqrt(sampled_elements)",
        "loss_tolerance_source": str(args.loss_tolerance_json.resolve()),
        "loss_tolerance_source_sha256": sha256(args.loss_tolerance_json),
        "require_route_hashes": args.require_route_hashes,
        "run_count_dimensionless": len(runs),
        "groups": groups,
        "nccl_training_self_repeat": nccl_training_self_repeat,
        "figures": figures,
        "combined_iteration_metrics": {
            "path": combined_csv.name,
            "sha256": sha256(combined_csv),
        },
    }
    comparison_path = args.output_dir / "comparison.json"
    comparison_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    write_markdown(document, args.output_dir / "comparison.md")
    print(json.dumps(document, indent=2, sort_keys=True))
    if args.fail_on_invalid and document["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
