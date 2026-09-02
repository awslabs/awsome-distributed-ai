#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
parse_results.py - Parse Megatron-DeepSpeed training logs into benchmark JSON.

Reads Slurm log files, extracts per-step metrics, and produces JSON files
matching the existing benchmark-results schema at:
  s3://<YOUR_BUCKET>/benchmark-results/<instance_type>/

Usage:
    python parse_results.py [--logs-dir logs] [--output-dir sweep_results]
    python parse_results.py --log-file logs/sweep_01_baseline_123.out --config-name 01_baseline
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
from datetime import datetime, timezone


# ============================================================
# Megatron-DeepSpeed log line patterns
# ============================================================
# Example: " iteration       10/      50 | consumed samples: ..."
# Example: "elapsed time per iteration (ms): 4725.7 | ..."
# Example: "lm loss: 1.3389E+01 | ..."
# Example: "learning rate: 3.000E-05 | ..."
# Example: "global batch size:   128 | ..."
# Example: "loss scale: 1.0 | ..."
# Example: "grad norm: 74.776 | ..."
# Example: "TFLOPs: 125.4 | ..."

ITER_PATTERN = re.compile(r"iteration\s+(\d+)/\s*(\d+)")
ELAPSED_PATTERN = re.compile(r"elapsed time per iteration \(ms\):\s*([\d.]+)")
LOSS_PATTERN = re.compile(r"lm loss:\s*([\d.eE+\-]+)")
LR_PATTERN = re.compile(r"learning rate:\s*([\d.eE+\-]+)")
GBS_PATTERN = re.compile(r"global batch size:\s*(\d+)")
LOSS_SCALE_PATTERN = re.compile(r"loss scale:\s*([\d.eE+\-]+)")
GRAD_NORM_PATTERN = re.compile(r"grad norm:\s*([\d.eE+\-]+)")
TFLOPS_PATTERN = re.compile(r"TFLOPs:\s*([\d.]+)")


def parse_log_file(log_path):
    """Parse a single Megatron-DeepSpeed log file and extract per-step metrics."""
    steps = []
    current_step = {}

    with open(log_path, "r") as f:
        for line in f:
            # Check for iteration marker
            m = ITER_PATTERN.search(line)
            if m:
                if current_step:
                    steps.append(current_step)
                current_step = {
                    "step": int(m.group(1)),
                    "total_steps": int(m.group(2)),
                }

            if not current_step:
                continue

            # Extract metrics from the same log block
            m = ELAPSED_PATTERN.search(line)
            if m:
                elapsed_ms = float(m.group(1))
                current_step["elapsed_ms"] = elapsed_ms
                current_step["step_time_s"] = round(elapsed_ms / 1000.0, 2)

            m = LOSS_PATTERN.search(line)
            if m:
                current_step["lm_loss"] = float(m.group(1))

            m = LR_PATTERN.search(line)
            if m:
                current_step["learning_rate"] = float(m.group(1))

            m = GBS_PATTERN.search(line)
            if m:
                current_step["global_batch_size"] = int(m.group(1))

            m = LOSS_SCALE_PATTERN.search(line)
            if m:
                current_step["loss_scale"] = float(m.group(1))

            m = GRAD_NORM_PATTERN.search(line)
            if m:
                current_step["grad_norm"] = float(m.group(1))

            m = TFLOPS_PATTERN.search(line)
            if m:
                current_step["tflops_per_gpu"] = float(m.group(1))

    # Don't forget the last step
    if current_step:
        steps.append(current_step)

    return steps


def compute_tflops_from_step_time(
    step_time_s,
    global_batch_size,
    seq_length=2048,
    hidden_size=12288,
    num_layers=80,
    num_heads=96,
    total_gpus=64,
):
    """
    Estimate TFLOPS/GPU for a GPT model using the standard formula:
    FLOPs per iteration = 8 * seq * hidden^2 * layers * (1 + seq/(6*hidden) + vocab/(12*hidden*layers))
    Simplified: ~= 8 * B * s * h^2 * L * (1 + s/(6h))
    where B = global_batch_size
    """
    vocab_size = 50257  # GPT-2 vocab
    s = seq_length
    h = hidden_size
    L = num_layers
    B = global_batch_size

    # Standard approximation for GPT FLOP count
    flops_per_iter = (
        8 * B * s * h * h * L * (1 + s / (6 * h) + vocab_size / (12 * h * L))
    )
    tflops_per_gpu = flops_per_iter / (step_time_s * total_gpus * 1e12)
    return round(tflops_per_gpu, 1)


def build_result_json(
    steps,
    config_name,
    job_id,
    nodes=8,
    gpus_per_node=8,
    tp=8,
    pp=2,
    zero_stage=1,
    mbs=1,
    gbs=64,
    seq_length=2048,
    precision="bf16",
    cluster="unknown",
    instance_type="unknown",
):
    """Build the benchmark JSON matching the existing schema."""
    total_gpus = nodes * gpus_per_node
    warmup_steps = 5
    total_steps = len(steps)

    # Ensure TFLOPS values exist (compute if not in logs)
    for step in steps:
        if "tflops_per_gpu" not in step and "step_time_s" in step:
            step["tflops_per_gpu"] = compute_tflops_from_step_time(
                step["step_time_s"],
                step.get("global_batch_size", gbs),
                seq_length=seq_length,
                total_gpus=total_gpus,
            )

    # Steady-state metrics (skip warmup)
    steady_steps = [s for s in steps if s.get("step", 0) > warmup_steps]

    if not steady_steps:
        print(f"Warning: No steady-state steps found for {config_name}")
        steady_steps = steps

    steady_tflops = [s["tflops_per_gpu"] for s in steady_steps if "tflops_per_gpu" in s]
    steady_times = [s["step_time_s"] for s in steady_steps if "step_time_s" in s]

    summary = {
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "steady_state_steps": len(steady_steps),
    }

    if steady_tflops:
        summary.update(
            {
                "steady_state_avg_tflops_per_gpu": round(
                    statistics.mean(steady_tflops), 2
                ),
                "steady_state_median_tflops_per_gpu": round(
                    statistics.median(steady_tflops), 1
                ),
                "steady_state_min_tflops_per_gpu": round(min(steady_tflops), 1),
                "steady_state_max_tflops_per_gpu": round(max(steady_tflops), 1),
                "steady_state_stdev_tflops_per_gpu": round(
                    statistics.stdev(steady_tflops), 2
                )
                if len(steady_tflops) > 1
                else 0.0,
                "peak_tflops_per_gpu": round(max(steady_tflops), 1),
            }
        )

    if steady_times:
        summary.update(
            {
                "steady_state_avg_step_time_s": round(statistics.mean(steady_times), 4),
                "steady_state_median_step_time_s": round(
                    statistics.median(steady_times), 2
                ),
                "steady_state_min_step_time_s": round(min(steady_times), 2),
                "steady_state_max_step_time_s": round(max(steady_times), 2),
            }
        )

    if steps:
        summary["final_loss"] = steps[-1].get("lm_loss", None)
        summary["initial_loss"] = steps[0].get("lm_loss", None)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    result = {
        "metadata": {
            "timestamp": timestamp,
            "job_id": str(job_id),
            "cluster": cluster,
            "instance_type": instance_type,
            "nodes": nodes,
            "gpus_per_node": gpus_per_node,
            "total_gpus": total_gpus,
            "model": "deepspeed-gpt-103b",
            "precision": precision,
            "framework": "megatron-deepspeed",
            "sweep_config": {
                "config_name": config_name,
                "tp": tp,
                "pp": pp,
                "zero_stage": zero_stage,
                "micro_batch_size": mbs,
                "global_batch_size": gbs,
                "seq_length": seq_length,
            },
        },
        "summary": summary,
        "steps": steps,
    }

    return result


def parse_sweep_jobs(
    jobs_csv, logs_dir, output_dir, cluster="unknown", instance_type="unknown"
):
    """Parse all jobs from the sweep tracking CSV."""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    with open(jobs_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            job_id = row["job_id"]
            config_name = row["config_name"]

            # Find the log file for this job
            log_pattern = f"sweep_{config_name}_{job_id}.out"
            log_path = os.path.join(logs_dir, log_pattern)

            if not os.path.exists(log_path):
                # Try alternate pattern
                log_candidates = [
                    f
                    for f in os.listdir(logs_dir)
                    if job_id in f and f.endswith(".out")
                ]
                if log_candidates:
                    log_path = os.path.join(logs_dir, log_candidates[0])
                else:
                    print(
                        f"Warning: No log file found for job {job_id} ({config_name})"
                    )
                    continue

            print(f"Parsing {config_name} (job {job_id}): {log_path}")
            steps = parse_log_file(log_path)

            if not steps:
                print(f"  Warning: No steps found in log file")
                continue

            result = build_result_json(
                steps=steps,
                config_name=config_name,
                job_id=job_id,
                tp=int(row.get("tp", 8)),
                pp=int(row.get("pp", 2)),
                zero_stage=int(row.get("zero", 1)),
                mbs=int(row.get("mbs", 1)),
                gbs=int(row.get("gbs", 64)),
                seq_length=int(row.get("seq_length", 2048)),
                cluster=cluster,
                instance_type=instance_type,
            )

            # Write individual JSON file
            now = datetime.now(timezone.utc)
            filename = (
                f"training_bench_deepspeed-gpt-103b_bf16_"
                f"{now.strftime('%Y-%m-%d_%H%M')}_job{job_id}.json"
            )
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as jf:
                json.dump(result, jf, indent=2)
            print(f"  Wrote: {filepath}")

            results.append(result)

    # Write combined summary
    summary_path = os.path.join(output_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote combined summary: {summary_path}")

    return results


def parse_single_log(
    log_file, config_name, output_dir, cluster="unknown", instance_type="unknown"
):
    """Parse a single log file."""
    os.makedirs(output_dir, exist_ok=True)

    # Extract job ID from filename
    job_id_match = re.search(r"_(\d+)\.out", log_file)
    job_id = job_id_match.group(1) if job_id_match else "unknown"

    print(f"Parsing {config_name} (job {job_id}): {log_file}")
    steps = parse_log_file(log_file)

    if not steps:
        print("Error: No steps found in log file")
        sys.exit(1)

    result = build_result_json(
        steps=steps,
        config_name=config_name,
        job_id=job_id,
        cluster=cluster,
        instance_type=instance_type,
    )

    now = datetime.now(timezone.utc)
    filename = (
        f"training_bench_deepspeed-gpt-103b_bf16_"
        f"{now.strftime('%Y-%m-%d_%H%M')}_job{job_id}.json"
    )
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote: {filepath}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Parse Megatron-DeepSpeed logs into benchmark JSON"
    )
    parser.add_argument(
        "--logs-dir", default="logs", help="Directory containing Slurm log files"
    )
    parser.add_argument(
        "--output-dir", default="sweep_results", help="Directory to write JSON results"
    )
    parser.add_argument(
        "--jobs-csv",
        default="sweep_results/sweep_jobs.csv",
        help="CSV file tracking sweep job IDs",
    )
    parser.add_argument(
        "--log-file", default=None, help="Parse a single log file instead of sweep CSV"
    )
    parser.add_argument(
        "--config-name",
        default="single_run",
        help="Config name for single log file parsing",
    )
    parser.add_argument(
        "--cluster",
        default=os.environ.get("CLUSTER_NAME", "unknown"),
        help="Cluster name for metadata (default: $CLUSTER_NAME or 'unknown')",
    )
    parser.add_argument(
        "--instance-type",
        default=os.environ.get("INSTANCE_TYPE", "unknown"),
        help="Instance type for metadata (default: $INSTANCE_TYPE or 'unknown')",
    )

    args = parser.parse_args()

    if args.log_file:
        parse_single_log(
            args.log_file,
            args.config_name,
            args.output_dir,
            cluster=args.cluster,
            instance_type=args.instance_type,
        )
    else:
        if not os.path.exists(args.jobs_csv):
            print(f"Error: Jobs CSV not found: {args.jobs_csv}")
            print(
                "Run sweep_runner.sh first, or use --log-file for single file parsing"
            )
            sys.exit(1)
        parse_sweep_jobs(
            args.jobs_csv,
            args.logs_dir,
            args.output_dir,
            cluster=args.cluster,
            instance_type=args.instance_type,
        )


if __name__ == "__main__":
    main()
