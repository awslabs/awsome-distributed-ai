#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""nsys_analyze.py — Automated Nsight Systems report analysis for distributed training.

Runs nsys stats on .nsys-rep files, categorizes GPU kernels, identifies bottlenecks,
and generates a structured Markdown report.

Usage:
    python nsys_analyze.py --reports /path/to/report1.nsys-rep [report2.nsys-rep ...]
    python nsys_analyze.py --reports /path/to/reports/  # directory of .nsys-rep files
    python nsys_analyze.py --reports report.nsys-rep --nsys-bin /path/to/nsys
    python nsys_analyze.py --reports report.nsys-rep --output /tmp/analysis.md

Requires: nsys binary accessible (auto-detected or via --nsys-bin)
"""

import argparse
import csv
import io
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("nsys_analyze")


# ── Kernel Classification ─────────────────────────────────────────────────────

KERNEL_CATEGORIES = {
    "NCCL AllGather": [r"ncclDevKernel_AllGather"],
    "NCCL ReduceScatter": [r"ncclDevKernel_ReduceScatter"],
    "NCCL AllReduce": [r"ncclDevKernel_AllReduce"],
    "NCCL SendRecv": [r"ncclDevKernel_SendRecv"],
    "NCCL Other": [r"ncclDevKernel", r"ncclKernel"],
    "GEMM (Compute)": [r"gemm", r"cutlass.*gemm", r"ampere.*gemm", r"hopper.*gemm", r"Gemm"],
    "Flash Attention": [r"flash.*fwd", r"flash.*bwd", r"flash.*kernel"],
    "Softmax": [r"SoftMax", r"softmax"],
    "Elementwise": [r"elementwise_kernel", r"vectorized_elementwise"],
    "Reduction": [r"reduce_kernel"],
    "Memory Copy": [r"memcpy", r"MemCpy", r"CatArray"],
    "Optimizer": [r"multi_tensor_apply", r"FusedAdam", r"adam"],
    "Embedding": [r"indexSelect", r"embedding", r"EmbeddingBag"],
    "Loss": [r"nll_loss", r"cross_entropy"],
    "LayerNorm/RMSNorm": [r"layer_norm", r"rms_norm", r"LayerNorm", r"RmsNorm"],
    "Activation": [r"silu", r"gelu", r"relu", r"SiLU", r"GeLU", r"ReLU"],
    "Triton": [r"triton"],
}


def classify_kernel(name: str) -> str:
    """Classify a CUDA kernel name into a category."""
    for category, patterns in KERNEL_CATEGORIES.items():
        for pattern in patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return category
    return "Other"


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class KernelStat:
    name: str
    time_pct: float
    total_time_ns: int
    instances: int
    avg_ns: float
    med_ns: float
    min_ns: int
    max_ns: int
    stddev_ns: float
    category: str = ""


@dataclass
class ApiStat:
    name: str
    time_pct: float
    total_time_ns: int
    num_calls: int
    avg_ns: float
    med_ns: float
    min_ns: int
    max_ns: int
    stddev_ns: float


@dataclass
class MemStat:
    operation: str
    time_pct: float = 0.0
    total_time_ns: int = 0
    count: int = 0
    avg_ns: float = 0.0
    total_mb: float = 0.0
    avg_mb: float = 0.0


@dataclass
class ReportAnalysis:
    """Analysis results for a single .nsys-rep file."""
    report_path: str
    kernel_stats: List[KernelStat] = field(default_factory=list)
    api_stats: List[ApiStat] = field(default_factory=list)
    mem_time_stats: List[MemStat] = field(default_factory=list)
    mem_size_stats: List[MemStat] = field(default_factory=list)
    osrt_stats: List[ApiStat] = field(default_factory=list)
    nvtx_stats: List[dict] = field(default_factory=list)
    category_summary: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # category -> (pct, total_s)
    bottleneck_type: str = ""
    recommendations: List[str] = field(default_factory=list)


# ── nsys stats runner ─────────────────────────────────────────────────────────

def find_nsys(nsys_bin: Optional[str] = None) -> str:
    """Find nsys binary."""
    if nsys_bin and os.path.isfile(nsys_bin):
        return nsys_bin
    # Search common paths (newest version first via reverse sort)
    import glob
    search = []
    search += sorted(glob.glob("/opt/nvidia/nsight-systems/*/target-linux-x64/nsys"), reverse=True)
    search += sorted(glob.glob("/opt/nvidia/nsight-systems/*/bin/nsys"), reverse=True)
    search += sorted(glob.glob("/nsight/*/target-linux-x64/nsys"), reverse=True)
    search += sorted(glob.glob("/nsight/*/bin/nsys"), reverse=True)
    for p in search:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    # Fall back to PATH
    import shutil
    nsys_path = shutil.which("nsys")
    if nsys_path:
        return nsys_path
    raise FileNotFoundError("nsys binary not found. Set --nsys-bin or add nsys to PATH.")


def run_nsys_stats(nsys_bin: str, report_path: str, report_name: str) -> str:
    """Run nsys stats for a specific report type and return CSV output."""
    cmd = [
        nsys_bin, "stats",
        "--report", report_name,
        "--format", "csv",
        "--output", "-",  # stdout
        "--force-export=true",
        report_path,
    ]
    logger.debug(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.warning(f"nsys stats {report_name} failed: {result.stderr.strip()}")
            return ""
        return result.stdout
    except subprocess.TimeoutExpired:
        logger.warning(f"nsys stats {report_name} timed out")
        return ""
    except FileNotFoundError:
        logger.error(f"nsys binary not found at {nsys_bin}")
        return ""


def parse_csv_output(raw_output: str) -> List[dict]:
    """Parse nsys stats CSV output, skipping header lines."""
    lines = raw_output.strip().split("\n")
    # Find the CSV header (starts after any non-CSV preamble)
    csv_start = 0
    for i, line in enumerate(lines):
        # nsys stats CSV headers start with a quoted field or a field name
        if line.startswith('"') or (i > 0 and ',' in line and not line.startswith(' ')):
            csv_start = i
            break
    if csv_start >= len(lines):
        return []
    csv_text = "\n".join(lines[csv_start:])
    reader = csv.DictReader(io.StringIO(csv_text))
    return [row for row in reader]


def safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val.replace(",", "").strip())
    except (ValueError, AttributeError):
        return default


def safe_int(val: str, default: int = 0) -> int:
    try:
        return int(float(val.replace(",", "").strip()))
    except (ValueError, AttributeError):
        return default


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_report(nsys_bin: str, report_path: str) -> ReportAnalysis:
    """Analyze a single .nsys-rep file."""
    analysis = ReportAnalysis(report_path=report_path)
    logger.info(f"Analyzing: {report_path}")

    # 1. CUDA GPU Kernel Summary
    raw = run_nsys_stats(nsys_bin, report_path, "cuda_gpu_kern_sum")
    if raw:
        for row in parse_csv_output(raw):
            name = row.get("Name", row.get("name", ""))
            ks = KernelStat(
                name=name,
                time_pct=safe_float(row.get("Time (%)", row.get("Time(%)", "0"))),
                total_time_ns=safe_int(row.get("Total Time (ns)", row.get("Total Time(ns)", "0"))),
                instances=safe_int(row.get("Instances", "0")),
                avg_ns=safe_float(row.get("Avg (ns)", row.get("Avg(ns)", "0"))),
                med_ns=safe_float(row.get("Med (ns)", row.get("Med(ns)", "0"))),
                min_ns=safe_int(row.get("Min (ns)", row.get("Min(ns)", "0"))),
                max_ns=safe_int(row.get("Max (ns)", row.get("Max(ns)", "0"))),
                stddev_ns=safe_float(row.get("StdDev (ns)", row.get("StdDev(ns)", "0"))),
                category=classify_kernel(name),
            )
            analysis.kernel_stats.append(ks)

    # 2. CUDA API Summary
    raw = run_nsys_stats(nsys_bin, report_path, "cuda_api_sum")
    if raw:
        for row in parse_csv_output(raw):
            api = ApiStat(
                name=row.get("Name", row.get("name", "")),
                time_pct=safe_float(row.get("Time (%)", "0")),
                total_time_ns=safe_int(row.get("Total Time (ns)", "0")),
                num_calls=safe_int(row.get("Num Calls", "0")),
                avg_ns=safe_float(row.get("Avg (ns)", "0")),
                med_ns=safe_float(row.get("Med (ns)", "0")),
                min_ns=safe_int(row.get("Min (ns)", "0")),
                max_ns=safe_int(row.get("Max (ns)", "0")),
                stddev_ns=safe_float(row.get("StdDev (ns)", "0")),
            )
            analysis.api_stats.append(api)

    # 3. CUDA GPU MemOps (by Time)
    raw = run_nsys_stats(nsys_bin, report_path, "cuda_gpu_mem_time_sum")
    if raw:
        for row in parse_csv_output(raw):
            ms = MemStat(
                operation=row.get("Operation", row.get("operation", "")),
                time_pct=safe_float(row.get("Time (%)", "0")),
                total_time_ns=safe_int(row.get("Total Time (ns)", "0")),
                count=safe_int(row.get("Count", "0")),
                avg_ns=safe_float(row.get("Avg (ns)", "0")),
            )
            analysis.mem_time_stats.append(ms)

    # 4. CUDA GPU MemOps (by Size)
    raw = run_nsys_stats(nsys_bin, report_path, "cuda_gpu_mem_size_sum")
    if raw:
        for row in parse_csv_output(raw):
            ms = MemStat(
                operation=row.get("Operation", row.get("operation", "")),
                count=safe_int(row.get("Count", "0")),
                total_mb=safe_float(row.get("Total (MB)", "0")),
                avg_mb=safe_float(row.get("Avg (MB)", "0")),
            )
            analysis.mem_size_stats.append(ms)

    # 5. OS Runtime Summary
    raw = run_nsys_stats(nsys_bin, report_path, "osrt_sum")
    if raw:
        for row in parse_csv_output(raw):
            api = ApiStat(
                name=row.get("Name", ""),
                time_pct=safe_float(row.get("Time (%)", "0")),
                total_time_ns=safe_int(row.get("Total Time (ns)", "0")),
                num_calls=safe_int(row.get("Num Calls", "0")),
                avg_ns=safe_float(row.get("Avg (ns)", "0")),
                med_ns=safe_float(row.get("Med (ns)", "0")),
                min_ns=safe_int(row.get("Min (ns)", "0")),
                max_ns=safe_int(row.get("Max (ns)", "0")),
                stddev_ns=safe_float(row.get("StdDev (ns)", "0")),
            )
            analysis.osrt_stats.append(api)

    # 6. NVTX Summary (if available)
    raw = run_nsys_stats(nsys_bin, report_path, "nvtx_sum")
    if raw:
        analysis.nvtx_stats = parse_csv_output(raw)

    # ── Compute category summary ─────────────────────────────────────────────
    total_kernel_time = sum(k.total_time_ns for k in analysis.kernel_stats)
    category_times: Dict[str, int] = {}
    for k in analysis.kernel_stats:
        cat = k.category
        category_times[cat] = category_times.get(cat, 0) + k.total_time_ns

    for cat, ns in sorted(category_times.items(), key=lambda x: -x[1]):
        pct = (ns / total_kernel_time * 100) if total_kernel_time > 0 else 0
        analysis.category_summary[cat] = (pct, ns / 1e9)

    # ── Classify bottleneck ──────────────────────────────────────────────────
    nccl_pct = sum(pct for cat, (pct, _) in analysis.category_summary.items() if "NCCL" in cat)
    compute_pct = sum(pct for cat, (pct, _) in analysis.category_summary.items()
                      if cat in ("GEMM (Compute)", "Flash Attention", "Softmax", "Triton"))
    memcpy_pct = sum(pct for cat, (pct, _) in analysis.category_summary.items() if "Memory" in cat)

    # Check CUDA API sync time
    sync_pct = 0.0
    for api in analysis.api_stats:
        if "Synchronize" in api.name:
            sync_pct += api.time_pct

    if nccl_pct > 40:
        analysis.bottleneck_type = "Communication Bound"
    elif sync_pct > 60:
        analysis.bottleneck_type = "Synchronization Bound"
    elif compute_pct > 70:
        analysis.bottleneck_type = "Compute Bound"
    elif memcpy_pct > 20:
        analysis.bottleneck_type = "Memory Transfer Bound"
    else:
        analysis.bottleneck_type = "Mixed / Balanced"

    # ── Generate recommendations ─────────────────────────────────────────────
    if nccl_pct > 30:
        # Check if using TCP/Socket
        analysis.recommendations.append(
            f"NCCL collectives consume {nccl_pct:.1f}% of GPU time. "
            "If using TCP/Socket transport, enable EFA with OFI-NCCL for 5-10x speedup. "
            "Set NCCL_NET=ofi and ensure aws-ofi-nccl plugin is in the container."
        )
        analysis.recommendations.append(
            "Consider increasing batch size to improve the compute-to-communication ratio."
        )

    # Check for H2D/D2H transfers (activation offloading indicator)
    h2d_mb = sum(m.total_mb for m in analysis.mem_size_stats if "Host-to-Device" in m.operation)
    d2h_mb = sum(m.total_mb for m in analysis.mem_size_stats if "Device-to-Host" in m.operation)
    if h2d_mb + d2h_mb > 50000:  # > 50 GB
        analysis.recommendations.append(
            f"Large Host<->Device transfers detected ({(h2d_mb + d2h_mb) / 1024:.1f} GB). "
            "This may indicate FSDP activation offloading. If GPU memory allows, "
            "try --offload_activations=0 to eliminate PCIe transfer overhead."
        )

    if sync_pct > 50:
        analysis.recommendations.append(
            f"CPU spends {sync_pct:.1f}% in CUDA synchronization calls. "
            "Avoid .item()/.cpu() in the training loop. Use loss.detach() for logging. "
            "Enable async data transfers with pin_memory=True and .to(non_blocking=True)."
        )

    # Check for large kernel stddev (stragglers)
    for k in analysis.kernel_stats[:10]:
        if k.stddev_ns > k.avg_ns * 0.5 and k.instances > 50:
            analysis.recommendations.append(
                f"High variance in '{k.name[:60]}...' "
                f"(avg={k.avg_ns / 1e6:.1f}ms, stddev={k.stddev_ns / 1e6:.1f}ms, {k.instances} calls). "
                "This may indicate stragglers or load imbalance."
            )
            break

    return analysis


# ── Report Generation ─────────────────────────────────────────────────────────

def generate_markdown_report(analyses: List[ReportAnalysis]) -> str:
    """Generate a comprehensive Markdown bottleneck report."""
    lines = []
    lines.append("# Nsight Systems Bottleneck Analysis Report")
    lines.append("")
    lines.append(f"**Reports analyzed:** {len(analyses)}")
    lines.append("")

    for i, a in enumerate(analyses):
        rank_label = Path(a.report_path).stem
        lines.append(f"## Worker: {rank_label}")
        lines.append("")
        lines.append(f"**Bottleneck Classification: {a.bottleneck_type}**")
        lines.append("")

        # Category breakdown
        lines.append("### GPU Kernel Time Breakdown")
        lines.append("")
        lines.append("| Category | % GPU Time | Total (s) | Top Kernel |")
        lines.append("|----------|-----------|-----------|------------|")

        for cat, (pct, total_s) in a.category_summary.items():
            if pct < 0.1:
                continue
            # Find top kernel in this category
            top_kern = ""
            for k in a.kernel_stats:
                if k.category == cat:
                    top_kern = k.name[:50]
                    break
            lines.append(f"| {cat} | {pct:.1f}% | {total_s:.1f}s | `{top_kern}` |")
        lines.append("")

        # CUDA API Summary (top 5)
        if a.api_stats:
            lines.append("### CUDA API Time (Top 5)")
            lines.append("")
            lines.append("| API | % CPU Time | Total (s) | Calls |")
            lines.append("|-----|-----------|-----------|-------|")
            for api in a.api_stats[:5]:
                lines.append(
                    f"| `{api.name}` | {api.time_pct:.1f}% | "
                    f"{api.total_time_ns / 1e9:.1f}s | {api.num_calls:,} |"
                )
            lines.append("")

        # Memory transfers
        if a.mem_size_stats:
            lines.append("### Memory Transfers")
            lines.append("")
            lines.append("| Direction | Total (GB) | Count | Avg (MB) |")
            lines.append("|-----------|-----------|-------|----------|")
            for m in a.mem_size_stats:
                if m.total_mb < 1:
                    continue
                lines.append(
                    f"| {m.operation} | {m.total_mb / 1024:.1f} | "
                    f"{m.count:,} | {m.avg_mb:.1f} |"
                )
            lines.append("")

        # NVTX ranges (if available)
        if a.nvtx_stats:
            lines.append("### PyTorch Operation Breakdown (NVTX)")
            lines.append("")
            lines.append("| Range | % Time | Total (ms) | Instances |")
            lines.append("|-------|--------|-----------|-----------|")
            for row in a.nvtx_stats[:15]:
                name = row.get("Range", row.get("range", ""))
                pct = safe_float(row.get("Time (%)", "0"))
                total = safe_float(row.get("Total Time (ns)", "0"))
                inst = safe_int(row.get("Instances", "0"))
                if pct < 0.1:
                    continue
                lines.append(f"| `{name[:50]}` | {pct:.1f}% | {total / 1e6:.1f} | {inst} |")
            lines.append("")

        # OS Runtime (top 3 by time)
        if a.osrt_stats:
            lines.append("### OS Runtime (Top 3)")
            lines.append("")
            for api in a.osrt_stats[:3]:
                lines.append(
                    f"- **{api.name}**: {api.time_pct:.1f}% "
                    f"({api.total_time_ns / 1e9:.1f}s, {api.num_calls:,} calls)"
                )
            lines.append("")

        # Recommendations
        if a.recommendations:
            lines.append("### Recommendations")
            lines.append("")
            for j, rec in enumerate(a.recommendations, 1):
                lines.append(f"{j}. {rec}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Cross-worker comparison (if multiple reports)
    if len(analyses) > 1:
        lines.append("## Cross-Worker Comparison")
        lines.append("")
        lines.append("| Worker | Bottleneck | NCCL % | Compute % | Sync % |")
        lines.append("|--------|-----------|--------|-----------|--------|")
        for a in analyses:
            label = Path(a.report_path).stem[:30]
            nccl = sum(p for c, (p, _) in a.category_summary.items() if "NCCL" in c)
            comp = sum(p for c, (p, _) in a.category_summary.items()
                       if c in ("GEMM (Compute)", "Flash Attention"))
            sync = sum(api.time_pct for api in a.api_stats if "Synchronize" in api.name)
            lines.append(f"| `{label}` | {a.bottleneck_type} | {nccl:.1f}% | {comp:.1f}% | {sync:.1f}% |")
        lines.append("")

    return "\n".join(lines)


def generate_json_report(analyses: List[ReportAnalysis]) -> str:
    """Generate a machine-readable JSON report."""
    data = []
    for a in analyses:
        entry = {
            "report": a.report_path,
            "bottleneck_type": a.bottleneck_type,
            "category_summary": {
                cat: {"pct": pct, "total_seconds": s}
                for cat, (pct, s) in a.category_summary.items()
            },
            "top_kernels": [
                {
                    "name": k.name[:100],
                    "category": k.category,
                    "time_pct": k.time_pct,
                    "total_ms": k.total_time_ns / 1e6,
                    "avg_ms": k.avg_ns / 1e6,
                    "instances": k.instances,
                }
                for k in a.kernel_stats[:20]
            ],
            "cuda_api_top5": [
                {
                    "name": api.name,
                    "time_pct": api.time_pct,
                    "total_s": api.total_time_ns / 1e9,
                    "calls": api.num_calls,
                }
                for api in a.api_stats[:5]
            ],
            "memory_transfers": [
                {
                    "operation": m.operation,
                    "total_gb": m.total_mb / 1024,
                    "count": m.count,
                    "avg_mb": m.avg_mb,
                }
                for m in a.mem_size_stats
            ],
            "recommendations": a.recommendations,
        }
        data.append(entry)
    return json.dumps(data, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Nsight Systems reports for distributed training bottlenecks"
    )
    parser.add_argument(
        "--reports", "-r", nargs="+", required=True,
        help="Path(s) to .nsys-rep files or a directory containing them"
    )
    parser.add_argument(
        "--nsys-bin", default=None,
        help="Path to nsys binary (auto-detected if not set)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output file path (default: stdout). Use .md for Markdown, .json for JSON"
    )
    parser.add_argument(
        "--format", "-f", choices=["markdown", "json", "both"], default="markdown",
        help="Output format (default: markdown)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Find nsys
    nsys_bin = find_nsys(args.nsys_bin)
    logger.info(f"Using nsys: {nsys_bin}")

    # Collect report files
    report_files = []
    for path in args.reports:
        p = Path(path)
        if p.is_dir():
            report_files.extend(sorted(p.glob("*.nsys-rep")))
        elif p.is_file() and p.suffix == ".nsys-rep":
            report_files.append(p)
        else:
            logger.warning(f"Skipping {path}: not a .nsys-rep file or directory")

    if not report_files:
        logger.error("No .nsys-rep files found")
        sys.exit(1)

    logger.info(f"Found {len(report_files)} report(s) to analyze")

    # Analyze each report
    analyses = []
    for rf in report_files:
        try:
            analysis = analyze_report(nsys_bin, str(rf))
            analyses.append(analysis)
        except Exception as e:
            logger.error(f"Failed to analyze {rf}: {e}")

    if not analyses:
        logger.error("No reports could be analyzed")
        sys.exit(1)

    # Generate output
    if args.format in ("markdown", "both"):
        md = generate_markdown_report(analyses)
        if args.output:
            out_path = args.output if args.output.endswith(".md") else args.output + ".md"
            Path(out_path).write_text(md)
            logger.info(f"Markdown report written to {out_path}")
        else:
            print(md)

    if args.format in ("json", "both"):
        js = generate_json_report(analyses)
        if args.output:
            out_path = args.output if args.output.endswith(".json") else args.output + ".json"
            Path(out_path).write_text(js)
            logger.info(f"JSON report written to {out_path}")
        elif args.format == "json":
            print(js)


if __name__ == "__main__":
    main()
