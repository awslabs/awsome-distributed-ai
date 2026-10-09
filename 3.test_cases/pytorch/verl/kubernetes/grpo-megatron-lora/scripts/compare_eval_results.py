#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# Compare Evaluation Results + Optional MLflow Logging
#
# Aggregates lm-eval-harness result JSONs and val-split JSONs into markdown
# comparison tables, and optionally logs the numbers to MLflow under the same
# run as the training job.
#
# Two modes:
#   1. SINGLE run vs. baseline (--run / --baseline).
#   2. LEARNING CURVE across multiple steps (--curve --steps 50,350,750
#      [--baseline <base-run>]). Renders one row per step with Δ vs. base.
#
# Expected on-disk layout (produced by submit_lmeval.sh + submit_val_eval.sh):
#
#   /fsx/data/verl/eval_results/<run>/
#     ├── lmeval/
#     │   └── results.json          # lm-eval-harness output (results.<task>.<metric>)
#     └── val_split.json            # eval_val_split.py output (summary.<data_source>)
#
# For --curve, <run> defaults to "<run-prefix>step<N>" per step (override the
# prefix with --run-prefix; default "qwen3-235b-").
#
# Usage (single):
#   python scripts/compare_eval_results.py \
#       --run qwen3-235b-step750 --baseline qwen3-235b-base \
#       --markdown-out /fsx/data/verl/eval_results/qwen3-235b-step750/comparison.md \
#       --mlflow-run-id <training-run-id>
#
# Usage (learning curve):
#   python scripts/compare_eval_results.py --curve \
#       --steps 50,350,750 --baseline qwen3-235b-base \
#       --run-prefix qwen3-235b- \
#       --markdown-out /fsx/data/verl/eval_results/qwen3-235b-curve.md \
#       --mlflow-run-id <training-run-id>
#
# MLflow logging (optional): requires MLFLOW_TRACKING_URI. With --mlflow-run-id,
# metrics attach to that existing run (appear alongside training). In --curve
# mode, per-step metrics are logged with the step number as the MLflow step so
# they render as a curve.
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# lm-eval task -> the metric keys we surface. lm-eval reports pass@1 (and pass@10
# when the harness is configured for it) under results.<task>.<metric>,<filter>.
#
# These are matched EXACTLY, not by prefix. kubernetes/lmeval-tasks/ ships humaneval_p4,
# humaneval_p10 and mbpp_p1/p4/p10, and LMEVAL_TASKS accepts a comma list -- so a prefix
# match collapsed every variant onto one key and silently kept whichever the JSON
# happened to iterate last. They are distinct benchmarks at distinct sample counts and
# must not be merged.
LMEVAL_TASKS = (
    "humaneval",
    "humaneval_p4",
    "humaneval_p10",
    "mbpp",
    "mbpp_p1",
    "mbpp_p4",
    "mbpp_p10",
)
# Metric name fragments lm-eval uses; we match on prefix to be robust to the
# ",create_test" / ",none" filter suffixes lm-eval appends.
LMEVAL_METRIC_PREFIXES = ("pass@1", "pass@10", "pass_at_1", "pass_at_10")


def _norm_metric(name: str) -> str:
    """Normalize an lm-eval metric key ('pass@1,create_test') -> 'pass_at_1'."""
    base = name.split(",")[0].strip()
    return base.replace("@", "_at_")


def _load_lmeval_metrics(results_dir: Path, run: str, strict: bool = True) -> dict:
    """Parse lm-eval results.json -> {'humaneval_pass_at_1': float, ...}.

    lm-eval shape:
      {"results": {"humaneval": {"pass@1,create_test": 0.42, "pass@1_stderr,...": ...},
                   "mbpp": {...}}, ...}

    Raises FileNotFoundError / ValueError when the run's results are absent or
    unreadable, so a missing input cannot render as a complete-looking report. Pass
    strict=False to treat a missing file as "not run" (used by --curve, where an
    intermediate step legitimately may not have been evaluated).
    """
    out: dict = {}
    path = results_dir / run / "lmeval" / "results.json"
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"no lm-eval results for run {run!r}: {path}")
        return out
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"could not read lm-eval results for run {run!r} ({path}): {exc}") from exc
    results = data.get("results", data)
    for task, metrics in results.items():
        # Exact task id. humaneval / humaneval_p4 / humaneval_p10 are separate
        # benchmarks and must keep separate metric keys.
        if task not in LMEVAL_TASKS or not isinstance(metrics, dict):
            continue
        for mkey, mval in metrics.items():
            if not isinstance(mval, (int, float)):
                continue
            # Keep stderr: docs/results.md quotes sigma and p-values throughout, and
            # dropping it made those claims unreproducible from the committed tooling.
            if "stderr" in mkey:
                base = _norm_metric(mkey.replace("_stderr", ""))
                if any(base.startswith(_norm_metric(pfx)) for pfx in LMEVAL_METRIC_PREFIXES):
                    out[f"{task}_{base}_stderr"] = float(mval)
                continue
            if any(mkey.startswith(p) for p in LMEVAL_METRIC_PREFIXES):
                out[f"{task}_{_norm_metric(mkey)}"] = float(mval)
    return out


def _load_val_split(results_dir: Path, run: str, strict: bool = True) -> dict:
    """Return dict of val-split metrics (per-data_source + __all__).

    An absent mean_score is reported as missing, NOT as 0.0. Coercing it to a float
    produced a real-looking measured zero that rendered as a large negative delta and
    was indistinguishable in the table from a genuine score of 0.
    """
    path = results_dir / run / "val_split.json"
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"no val-split results for run {run!r}: {path}")
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"could not read val-split results for run {run!r} ({path}): {exc}") from exc
    summary = data.get("summary", {})
    out = {}
    for ds, stats in summary.items():
        norm = ds.replace("/", "_").replace(".", "_")
        if "mean_score" in stats:
            out[f"valsplit_{norm}_mean_score"] = float(stats["mean_score"])
        for k, v in stats.items():
            if k.startswith("pass@"):
                out[f"valsplit_{norm}_{k.replace('@', '_at_')}"] = float(v)
        if "count" in stats:
            out[f"valsplit_{norm}_count"] = int(stats["count"])
        if "errors" in stats:
            out[f"valsplit_{norm}_errors"] = int(stats["errors"])
    return out


def _load_run(results_dir: Path, run: str, strict: bool = True) -> dict:
    """Load both instruments for one run.

    In strict mode a run must yield at least one metric: an empty result means the run
    name is wrong or the eval never completed, and reporting that as a table of dashes
    with exit 0 is how a bad input becomes a plausible-looking answer.
    """
    m = {}
    # A run may legitimately have only one of the two instruments, so each is tolerated
    # individually; the combined result being empty is not.
    errors = []
    for loader in (_load_lmeval_metrics, _load_val_split):
        try:
            m.update(loader(results_dir, run, strict=False))
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError("; ".join(errors))
    if strict and not m:
        raise FileNotFoundError(
            f"run {run!r} produced no metrics. Looked for\n"
            f"  {results_dir / run / 'lmeval' / 'results.json'}\n"
            f"  {results_dir / run / 'val_split.json'}\n"
            f"Check --results-dir and the run name."
        )
    return m


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, int):
        return str(v)
    return f"{v:.4f}"


def _pp(v_run, v_base):
    """Percentage-point delta string for pass@k (0..1 scaled)."""
    if v_run is None or v_base is None:
        return ""
    return f"{(v_run - v_base) * 100:+.2f} pp"


# ---- single run vs baseline ------------------------------------------------


def _build_markdown_single(run, baseline, run_metrics, base_metrics) -> str:
    lines = [f"# Evaluation Results: `{run}`", ""]
    if baseline:
        lines += [f"Comparison against baseline: `{baseline}`", ""]

    lines += ["## lm-evaluation-harness (HumanEval + MBPP)", ""]
    if baseline:
        lines += ["| Benchmark | Metric | Base | Checkpoint | Δ |", "|---|---|---:|---:|---:|"]
    else:
        lines += ["| Benchmark | Metric | Value |", "|---|---|---:|"]

    for task in LMEVAL_TASKS:
        for metric in ("pass_at_1", "pass_at_10"):
            key = f"{task}_{metric}"
            if key not in run_metrics and key not in base_metrics:
                continue
            v_run, v_base = run_metrics.get(key), base_metrics.get(key)
            pretty = metric.replace("pass_at_", "pass@")
            if baseline:
                lines.append(f"| {task} | {pretty} | {_fmt(v_base)} | {_fmt(v_run)} | {_pp(v_run, v_base)} |")
            else:
                lines.append(f"| {task} | {pretty} | {_fmt(v_run)} |")
    lines.append("")

    lines += ["## Validation-Split Rewards (mixed-code-math / val.parquet)", ""]
    if baseline:
        lines += ["| Data source | Metric | Base | Checkpoint | Δ |", "|---|---|---:|---:|---:|"]
    else:
        lines += ["| Data source | Metric | Value |", "|---|---|---:|"]

    val_prefixes = {
        k.removeprefix("valsplit_").rsplit("_mean_score", 1)[0]
        for k in list(run_metrics) + list(base_metrics)
        if k.startswith("valsplit_") and k.endswith("_mean_score")
    }
    for ds in sorted(val_prefixes):
        key = f"valsplit_{ds}_mean_score"
        v_run, v_base = run_metrics.get(key), base_metrics.get(key)
        if baseline:
            delta = f"{(v_run - v_base):+.4f}" if (v_run is not None and v_base is not None) else ""
            lines.append(f"| {ds} | mean_score | {_fmt(v_base)} | {_fmt(v_run)} | {delta} |")
        else:
            lines.append(f"| {ds} | mean_score | {_fmt(v_run)} |")
    lines.append("")
    return "\n".join(lines)


# ---- learning curve --------------------------------------------------------


def _curve_metric_keys(step_metrics: dict) -> list:
    """Ordered list of metric keys to show as columns in the curve table."""
    keys = []
    for task in LMEVAL_TASKS:
        for metric in ("pass_at_1", "pass_at_10"):
            keys.append(f"{task}_{metric}")
    # val-split mean_score per data source (union across all steps)
    val = sorted(
        {
            k
            for m in step_metrics.values()
            for k in m
            if k.startswith("valsplit_") and k.endswith("_mean_score")
        }
    )
    keys.extend(val)
    # keep only keys present in at least one step
    present = {k for m in step_metrics.values() for k in m}
    return [k for k in keys if k in present]


def _pretty_col(key: str) -> str:
    if key.startswith("valsplit_"):
        ds = key.removeprefix("valsplit_").rsplit("_mean_score", 1)[0]
        return f"val:{ds}"
    return key.replace("_pass_at_", " p@").replace("pass_at_", "p@")


def _build_markdown_curve(steps, step_metrics, baseline, base_metrics) -> str:
    lines = ["# Evaluation Learning Curve", ""]
    if baseline:
        lines += [f"Baseline: `{baseline}` (shown as step 0 / Δ reference)", ""]

    cols = _curve_metric_keys(step_metrics)
    if not cols:
        return "# Evaluation Learning Curve\n\n(no metrics found)\n"

    header = "| Step | " + " | ".join(_pretty_col(c) for c in cols) + " |"
    sep = "|---|" + "|".join("---:" for _ in cols) + "|"
    lines += [header, sep]

    if baseline:
        row = ["base"] + [_fmt(base_metrics.get(c)) for c in cols]
        lines.append("| " + " | ".join(row) + " |")

    for step in steps:
        m = step_metrics[step]
        row = [str(step)] + [_fmt(m.get(c)) for c in cols]
        lines.append("| " + " | ".join(row) + " |")

    # Δ vs base for the last step
    if baseline and steps:
        last = step_metrics[steps[-1]]
        drow = [f"Δ({steps[-1]}-base)"]
        for c in cols:
            v, b = last.get(c), base_metrics.get(c)
            if v is None or b is None:
                drow.append("—")
            elif "pass_at" in c or "p@" in _pretty_col(c):
                drow.append(f"{(v - b) * 100:+.2f}pp")
            else:
                drow.append(f"{(v - b):+.4f}")
        lines.append("| " + " | ".join(drow) + " |")

    lines.append("")
    return "\n".join(lines)


# ---- MLflow ----------------------------------------------------------------


def _log_mlflow_single(run_name, metrics, baseline_metrics, mlflow_run_id, experiment_name):
    mlflow = _mlflow_ready()
    if mlflow is None:
        return
    if mlflow_run_id:
        client = mlflow.tracking.MlflowClient()
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                client.log_metric(mlflow_run_id, f"eval/{run_name}/{k}", float(v))
        if baseline_metrics:
            for k, v in baseline_metrics.items():
                if isinstance(v, (int, float)):
                    client.log_metric(mlflow_run_id, f"eval/baseline/{k}", float(v))
        print(f"Logged eval metrics to existing MLflow run {mlflow_run_id}", flush=True)
    else:
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"eval-{run_name}"):
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"eval/{k}", float(v))
            if baseline_metrics:
                for k, v in baseline_metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"baseline/{k}", float(v))
        print(f"Logged eval metrics to new MLflow run 'eval-{run_name}'", flush=True)


def _log_mlflow_curve(steps, step_metrics, baseline_metrics, mlflow_run_id, experiment_name):
    mlflow = _mlflow_ready()
    if mlflow is None:
        return
    if mlflow_run_id:
        client = mlflow.tracking.MlflowClient()
        for step in steps:
            for k, v in step_metrics[step].items():
                if isinstance(v, (int, float)):
                    # step= makes these render as a curve in the MLflow UI
                    client.log_metric(mlflow_run_id, f"eval/{k}", float(v), step=int(step))
        if baseline_metrics:
            for k, v in baseline_metrics.items():
                if isinstance(v, (int, float)):
                    client.log_metric(mlflow_run_id, f"eval/baseline/{k}", float(v))
        print(f"Logged curve eval metrics to existing MLflow run {mlflow_run_id}", flush=True)
    else:
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="eval-curve"):
            for step in steps:
                for k, v in step_metrics[step].items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"eval/{k}", float(v), step=int(step))
        print("Logged curve eval metrics to new MLflow run 'eval-curve'", flush=True)


def _mlflow_ready():
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed — skipping MLflow logging", flush=True)
        return None
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        print("MLFLOW_TRACKING_URI not set — skipping MLflow logging", flush=True)
        return None
    mlflow.set_tracking_uri(uri)
    return mlflow


# ---- main ------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="Single-mode: run name (subdir of --results-dir)")
    p.add_argument("--curve", action="store_true", help="Learning-curve mode across --steps")
    p.add_argument("--steps", default=None, help="Curve mode: comma-separated steps, e.g. 50,350,750")
    p.add_argument("--run-prefix", default="qwen3-235b-", help="Curve mode: run dir = <prefix>step<N>")
    p.add_argument("--baseline", default=None, help="Baseline run name for comparison")
    p.add_argument("--results-dir", default="/fsx/data/verl/eval_results")
    p.add_argument("--markdown-out", default=None, help="Path to write markdown table")
    p.add_argument("--json-out", default=None, help="Path to write combined metrics JSON")
    p.add_argument("--mlflow-run-id", default=None, help="Existing MLflow run ID to attach metrics to")
    p.add_argument("--mlflow-experiment", default=None, help="MLflow experiment name (for new run)")
    args = p.parse_args()

    results_dir = Path(args.results_dir)

    def load(run: str) -> dict:
        try:
            return _load_run(results_dir, run)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(f"ERROR: {exc}") from exc

    base_metrics = {}
    if args.baseline:
        base_metrics = load(args.baseline)

    if args.curve:
        if not args.steps:
            p.error("--curve requires --steps (e.g. --steps 50,350,750)")
        steps = [int(s) for s in args.steps.split(",") if s.strip()]
        step_metrics = {s: load(f"{args.run_prefix}step{s}") for s in steps}
        md = _build_markdown_curve(steps, step_metrics, args.baseline, base_metrics)
        print(md, flush=True)
        _write_outputs(args, md, {"steps": step_metrics, "baseline": base_metrics})
        _log_mlflow_curve(steps, step_metrics, base_metrics if args.baseline else None,
                          args.mlflow_run_id, args.mlflow_experiment)
        return

    if not args.run:
        p.error("provide --run (single mode) or --curve --steps ... (curve mode)")
    run_metrics = load(args.run)
    md = _build_markdown_single(args.run, args.baseline, run_metrics, base_metrics)
    print(md, flush=True)
    _write_outputs(args, md, {"run": run_metrics, "baseline": base_metrics})
    _log_mlflow_single(args.run, run_metrics, base_metrics if args.baseline else None,
                       args.mlflow_run_id, args.mlflow_experiment)


def _write_outputs(args, md: str, combined: dict):
    if args.markdown_out:
        md_path = Path(args.markdown_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md)
        print(f"\nWrote markdown to {md_path}", flush=True)
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(combined, indent=2))
        print(f"Wrote combined JSON to {json_path}", flush=True)


if __name__ == "__main__":
    main()
