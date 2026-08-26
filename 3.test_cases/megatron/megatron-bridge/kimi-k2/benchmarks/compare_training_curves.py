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


def loss_tolerance(path: Path) -> float:
    payload = json.loads(path.read_text())
    tolerance = payload.get("tolerance", payload)
    value = float(tolerance["loss"])
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"invalid loss tolerance in {path}: {value}")
    return value


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
    group: list[Run], tolerance: float, require_route_hashes: bool
) -> dict:
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
        run_results[run.arm] = {
            "artifact_path": str(run.path),
            "environment_sha256": sha256(run.path / "environment.txt"),
            "iterations_observed_dimensionless": len(run.records),
            "iterations_expected_dimensionless": expected,
            "iteration_metrics_complete": complete and required_fields,
            "all_metrics_finite": finite,
            "no_skipped_or_nan_iterations": no_skips,
            "learning_rate_matches_environment": learning_rate_matches,
            "route_hashes_observed": len(run.route_hashes),
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
        for arm, run in by_arm.items():
            if arm == "nccl-alltoall":
                continue
            deltas = []
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
                        "within_nccl_self_repeat_tolerance": delta <= tolerance,
                    }
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
        for result in run_results.values()
    )
    comparison_pass = bool(comparisons) and all(
        item["loss_curve_within_nccl_self_repeat_tolerance"]
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


def plot_cell(cell: str, runs: list[Run], output_dir: Path) -> list[str]:
    repeats = sorted({run.repeat for run in runs})
    figure, axes = plt.subplots(
        len(repeats),
        1,
        figsize=(9.2, 3.7 * len(repeats)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
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
                marker="o" if len(iterations) <= 8 else None,
                markersize=3.5,
            )
        if (
            all_losses
            and min(all_losses) > 0.0
            and max(all_losses) / min(all_losses) >= 10.0
        ):
            axis.set_yscale("log")
        axis.set_title(f"Repeat {repeat} (dimensionless)")
        axis.set_ylabel("LM loss (dimensionless)")
        axis.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("Optimizer iteration (dimensionless)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    figure.suptitle(f"Kimi-K2 EP arm loss curves: {cell}", fontsize=14)
    stem = f"loss-curves-{safe_name(cell)}"
    paths = []
    for suffix in ("svg", "png"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=180)
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
        "",
    ]
    for group in document["groups"]:
        lines.extend(
            [
                f"## {group['cell']}, repeat {group['repeat_dimensionless']} (dimensionless)",
                "",
                f"Group status: `{group['status']}`.",
                "",
                "| Arm | Iterations, dimensionless | First loss, dimensionless | Last loss, dimensionless | Maximum absolute loss delta vs NCCL, dimensionless | Curve gate | Route hash gate |",
                "|---|---:|---:|---:|---:|---|---|",
            ]
        )
        for arm in ARMS:
            run = group["runs"].get(arm)
            if run is None:
                continue
            comparison = group["comparisons_vs_nccl_alltoall"].get(arm)
            max_delta = (
                "Reference"
                if arm == "nccl-alltoall"
                else f"{comparison['max_absolute_loss_delta_dimensionless']:.10g}"
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
            lines.append(
                f"| `{arm}` | {run['iterations_observed_dimensionless']} | "
                f"{run['loss_first_dimensionless']:.10g} | {run['loss_last_dimensionless']:.10g} | "
                f"{max_delta} | {curve_gate} | {route_gate} |"
            )
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--loss-tolerance-json", required=True, type=Path)
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
    tolerance = loss_tolerance(args.loss_tolerance_json)
    grouped: dict[tuple[str, int], list[Run]] = {}
    for run in runs:
        grouped.setdefault((run.cell, run.repeat), []).append(run)
    groups = [
        compare_group(group, tolerance, args.require_route_hashes)
        for _, group in sorted(grouped.items())
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = args.output_dir / "iteration-metrics.csv"
    write_combined_csv(runs, combined_csv)
    figures = {
        cell: plot_cell(
            cell, [run for run in runs if run.cell == cell], args.output_dir
        )
        for cell in sorted({run.cell for run in runs})
    }
    document = {
        "schema_version": 1,
        "status": "PASS"
        if all(group["status"] == "PASS" for group in groups)
        else "FAIL",
        "loss_tolerance_dimensionless": tolerance,
        "loss_tolerance_source": str(args.loss_tolerance_json.resolve()),
        "loss_tolerance_source_sha256": sha256(args.loss_tolerance_json),
        "require_route_hashes": args.require_route_hashes,
        "run_count_dimensionless": len(runs),
        "groups": groups,
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
