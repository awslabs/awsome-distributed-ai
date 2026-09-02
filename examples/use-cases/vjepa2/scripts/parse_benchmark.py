#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Parse V-JEPA 2 / V-JEPA 2.1 training logs and compute benchmark metrics.

V-JEPA 2/2.1 logs lines like:
  [epoch, itr] loss: 0.123 masks: [...] [wd: ...] [lr: ...] [mem: ...] [iter: 1234.5 ms] [gpu: 1200.0 ms] [data: 34.5 ms]

Usage:
    python parse_benchmark.py --log_file /path/to/training.log \
        --warmup_iters 20 \
        --batch_size_per_gpu 24 \
        --num_gpus 64 \
        --gpu_type h200 \
        --model_name "V-JEPA 2.1"
"""

import argparse
import re
import sys

# GPU specifications for supported instance types
GPU_SPECS = {
    "h200": {
        "name": "H200",
        "mem_gb": 141,
        "instance": "p5en.48xlarge",
        "peak_bf16_tflops": 989.4,
    },
    "b200": {
        "name": "B200",
        "mem_gb": 178,
        "instance": "p6-b200.48xlarge",
        "peak_bf16_tflops": 2250.0,
    },
}


def parse_log_file(log_file):
    """Extract iteration metrics from V-JEPA 2 training log."""
    pattern = re.compile(
        r"\[(\d+),\s*(\d+)\]\s+loss:\s+([\d.]+).*"
        r"\[mem:\s+([\d.e+]+)\].*"
        r"\[iter:\s+([\d.]+)\s+ms\].*"
        r"\[gpu:\s+([\d.]+)\s+ms\].*"
        r"\[data:\s+([\d.]+)\s+ms\]"
    )
    entries = []
    with open(log_file, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                entries.append(
                    {
                        "epoch": int(m.group(1)),
                        "itr": int(m.group(2)),
                        "loss": float(m.group(3)),
                        "mem_mb": float(m.group(4)),
                        "iter_ms": float(m.group(5)),
                        "gpu_ms": float(m.group(6)),
                        "data_ms": float(m.group(7)),
                    }
                )
    return entries


def compute_metrics(
    entries,
    warmup_iters,
    batch_size_per_gpu,
    num_gpus,
    patches_per_sample,
):
    """Compute throughput from parsed log entries."""
    # Skip warmup iterations
    steady = [e for e in entries if e["itr"] >= warmup_iters]
    if not steady:
        print(
            f"No entries found after warmup ({warmup_iters} iters). "
            f"Total entries: {len(entries)}",
            file=sys.stderr,
        )
        sys.exit(1)

    avg_iter_ms = sum(e["iter_ms"] for e in steady) / len(steady)
    avg_gpu_ms = sum(e["gpu_ms"] for e in steady) / len(steady)
    avg_data_ms = sum(e["data_ms"] for e in steady) / len(steady)
    avg_loss = sum(e["loss"] for e in steady) / len(steady)
    max_mem_mb = max(e["mem_mb"] for e in steady)

    avg_iter_sec = avg_iter_ms / 1000.0
    global_batch_size = batch_size_per_gpu * num_gpus

    samples_per_sec = global_batch_size / avg_iter_sec
    patches_per_sec = samples_per_sec * patches_per_sample

    # Note: MFU is not computed. V-JEPA's masking architecture makes FLOP
    # counting non-trivial -- the context encoder processes only visible
    # tokens (~15% of the sequence) while the target encoder processes all
    # tokens in forward-only mode (no backward pass). Samples/sec is the
    # primary throughput metric.

    return {
        "num_steady_iters": len(steady),
        "avg_iter_ms": avg_iter_ms,
        "avg_gpu_ms": avg_gpu_ms,
        "avg_data_ms": avg_data_ms,
        "avg_loss": avg_loss,
        "max_mem_mb": max_mem_mb,
        "max_mem_gb": max_mem_mb / 1024.0,
        "global_batch_size": global_batch_size,
        "samples_per_sec": samples_per_sec,
        "patches_per_sec": patches_per_sec,
    }


def print_results(metrics, batch_size_per_gpu, num_gpus, model_params, gpu_type, model_name):
    """Print benchmark results as a markdown table."""
    gpu = GPU_SPECS[gpu_type]
    print(f"\n## {model_name} Benchmark Results\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Model | {model_name} ViT-g/16 ({model_params / 1e9:.1f}B params) |")
    print(f"| Nodes | {num_gpus // 8} x {gpu['instance']} |")
    print(f"| GPUs | {num_gpus} x {gpu['name']} ({gpu['mem_gb']}GB) |")
    print(f"| Batch size (per GPU) | {batch_size_per_gpu} |")
    print(f"| Global batch size | {metrics['global_batch_size']} |")
    print(f"| Precision | BF16 |")
    print(f"| Steady-state iters | {metrics['num_steady_iters']} |")
    print(f"| Avg iter time | {metrics['avg_iter_ms']:.1f} ms |")
    print(f"| Avg GPU time | {metrics['avg_gpu_ms']:.1f} ms |")
    print(f"| Avg data load time | {metrics['avg_data_ms']:.1f} ms |")
    print(f"| Samples/sec | {metrics['samples_per_sec']:.1f} |")
    print(f"| Patches/sec | {metrics['patches_per_sec']:.0f} |")
    print(f"| Peak GPU memory | {metrics['max_mem_gb']:.1f} GB |")
    print(f"| Avg loss | {metrics['avg_loss']:.4f} |")
    print()


def main():
    parser = argparse.ArgumentParser(description="Parse V-JEPA 2/2.1 benchmark logs")
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--warmup_iters", type=int, default=20)
    parser.add_argument("--batch_size_per_gpu", type=int, default=24)
    parser.add_argument("--num_gpus", type=int, default=64)
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="h200",
        choices=list(GPU_SPECS.keys()),
        help="GPU type for output metadata (default: h200)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="V-JEPA 2",
        help="Model name for display in results (default: 'V-JEPA 2')",
    )
    parser.add_argument(
        "--model_params",
        type=float,
        default=1.1e9,
        help="Total model parameters (encoder + predictor), used for display only",
    )
    parser.add_argument(
        "--patches_per_sample",
        type=int,
        default=2048,
        help="Number of patches per video sample: (crop/patch)^2 * (frames/tubelet)",
    )
    args = parser.parse_args()

    entries = parse_log_file(args.log_file)
    if not entries:
        print(f"No metric entries found in {args.log_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(entries)} log entries from {args.log_file}")

    metrics = compute_metrics(
        entries,
        warmup_iters=args.warmup_iters,
        batch_size_per_gpu=args.batch_size_per_gpu,
        num_gpus=args.num_gpus,
        patches_per_sample=args.patches_per_sample,
    )

    print_results(
        metrics,
        args.batch_size_per_gpu,
        args.num_gpus,
        args.model_params,
        args.gpu_type,
        args.model_name,
    )


if __name__ == "__main__":
    main()
