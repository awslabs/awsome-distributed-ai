#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Parse PointWorld training logs to compute throughput and average loss.

PointWorld's Trainer prints per-step lines that include the global step, the
training loss, and a wall-clock timestamp. This script scans a captured log
file (e.g. ``kubectl logs pytorchjob/pointworld-pretrain-worker-0 > run.log``),
extracts per-step timing, and reports steady-state samples/sec across the
cluster after a warmup window.

Because exact log formatting can evolve upstream, the step/loss/time regexes are
configurable via flags. The defaults match the release Trainer's
``step=<N> ... loss=<F>`` style lines; adjust ``--step_regex`` / ``--loss_regex``
if your build differs.

Example
-------
    python parse_benchmark.py \
        --log_file run.log \
        --warmup_steps 20 \
        --global_batch_size 176 \
        --num_gpus 64 \
        --gpu_type h200
"""

import argparse
import re
import sys
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_file", required=True, help="Path to a captured training log.")
    parser.add_argument("--warmup_steps", type=int, default=20, help="Steps to skip before measuring steady state.")
    parser.add_argument(
        "--global_batch_size",
        type=int,
        required=True,
        help="Effective global batch size (per_gpu_batch_size * num_gpus).",
    )
    parser.add_argument("--num_gpus", type=int, required=True, help="Total GPUs in the run.")
    parser.add_argument("--gpu_type", default="h200", help="GPU label for the report (e.g. h200).")
    parser.add_argument(
        "--step_regex",
        default=r"B=(\d+)",
        help="Regex with one capture group for the global step counter. "
             "Default matches PointWorld's 'B=<batch>' log format.",
    )
    parser.add_argument(
        "--loss_regex",
        default=r"Loss=([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)",
        help="Regex with one capture group for the training loss. "
             "Default matches PointWorld's 'Loss=<value>' log format.",
    )
    parser.add_argument(
        "--time_regex",
        default=r"(?:iter_time|step_time|time)[=\s:]+([0-9]*\.?[0-9]+)",
        help="Regex with one capture group for per-step time in seconds. "
             "PointWorld logs do not include per-step time by default; "
             "throughput will be skipped unless this matches.",
    )
    parser.add_argument(
        "--timestamp_regex",
        default=r"^\[(\d{2}:\d{2}:\d{2})\s",
        help="Regex with one capture group for a wall-clock timestamp (HH:MM:SS) "
             "at the start of each step line. Used to derive per-step time when "
             "--time_regex does not match. Default matches PointWorld's "
             "'[HH:MM:SS run-name]' prefix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    step_re = re.compile(args.step_regex)
    loss_re = re.compile(args.loss_regex)
    time_re = re.compile(args.time_regex)
    ts_re = re.compile(args.timestamp_regex)

    steps, losses, times, timestamps = [], [], [], []
    try:
        with open(args.log_file, "r", errors="ignore") as fh:
            for line in fh:
                m_step = step_re.search(line)
                if not m_step:
                    continue
                step = int(m_step.group(1))
                m_loss = loss_re.search(line)
                m_time = time_re.search(line)
                m_ts = ts_re.search(line)
                steps.append(step)
                losses.append(float(m_loss.group(1)) if m_loss else float("nan"))
                if m_time:
                    times.append(float(m_time.group(1)))
                if m_ts:
                    timestamps.append(m_ts.group(1))
    except FileNotFoundError:
        sys.exit(f"Log file not found: {args.log_file}")

    if not steps:
        sys.exit(
            "No step lines matched. Inspect the log and adjust --step_regex "
            "(and --loss_regex/--time_regex) to match your build's format."
        )

    # Derive per-step times from consecutive timestamp diffs if no explicit time field.
    if not times and len(timestamps) > 1:
        fmt = "%H:%M:%S"
        derived = []
        for i in range(1, len(timestamps)):
            try:
                t0 = datetime.strptime(timestamps[i - 1], fmt)
                t1 = datetime.strptime(timestamps[i], fmt)
                diff = (t1 - t0).total_seconds()
                # Guard against midnight rollover and out-of-order lines.
                if 0 < diff < 3600:
                    derived.append(diff)
            except ValueError:
                pass
        if derived:
            times = derived

    n_warm = min(args.warmup_steps, max(len(steps) - 1, 0))
    measured_losses = [v for v in losses[n_warm:] if v == v]  # drop NaN
    measured_times = times[n_warm:] if times else []

    print("=" * 60)
    print("PointWorld Pre-training Benchmark Summary")
    print("=" * 60)
    print(f"  Log file            : {args.log_file}")
    print(f"  GPU type            : {args.gpu_type}")
    print(f"  Num GPUs            : {args.num_gpus}")
    print(f"  Global batch size   : {args.global_batch_size}")
    print(f"  Steps parsed        : {len(steps)} (warmup skipped: {n_warm})")
    if measured_losses:
        print(f"  Mean loss (steady)  : {sum(measured_losses) / len(measured_losses):.4f}")

    if measured_times:
        mean_step_time = sum(measured_times) / len(measured_times)
        samples_per_sec = args.global_batch_size / mean_step_time if mean_step_time > 0 else float("nan")
        print(f"  Mean step time (s)  : {mean_step_time:.4f}")
        print(f"  Throughput          : {samples_per_sec:.1f} samples/sec (cluster)")
        print(f"  Per-GPU throughput  : {samples_per_sec / args.num_gpus:.2f} samples/sec/GPU")
    else:
        print("  Per-step time not found in log; supply --time_regex to enable")
        print("  throughput calculation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
