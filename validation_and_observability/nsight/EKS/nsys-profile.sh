#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# nsys-profile.sh — Nsight Systems profiling wrapper for EKS distributed training
#
# This script wraps a training command with `nsys profile` using modern best practices.
# It is designed to be deployed as a ConfigMap and mounted into PyTorchJob pods.
#
# Usage:
#   nsys-profile.sh [--delay=N] [--duration=N] [--output-dir=DIR] [--ranks=0,1|all] -- <training_command>
#
# Or simply configure via environment variables and use:
#   nsys-profile.sh -- torchrun --nproc_per_node=8 ...
#
# Features:
#   - Auto-detects nsys binary on HyperPod and standard DLAMI paths
#   - PyTorch autograd NVTX annotations (--pytorch=autograd-shapes-nvtx)
#   - Python call stack sampling (--python-sampling)
#   - GPU hardware metrics (--gpu-metrics-devices) for supported GPUs
#   - CUDA memory tracking (--cuda-memory-usage)
#   - Selective rank profiling to minimize overhead at scale
#   - Proper --kill=none so training continues after profiling window ends
#
# Configuration via environment variables:
#   NSYS_DELAY            — Delay before collecting (seconds, default: 30)
#   NSYS_DURATION         — Collection duration (seconds, default: 90)
#   NSYS_OUTPUT_DIR       — Directory for .nsys-rep files (default: /local/nsight-reports)
#   NSYS_TRACE            — Trace APIs (default: cuda,nvtx,osrt)
#   NSYS_PYTORCH_MODE     — PyTorch NVTX mode (default: autograd-shapes-nvtx)
#   NSYS_PYTHON_SAMPLE    — Enable Python sampling (default: true)
#   NSYS_GPU_METRICS      — GPU HW metrics (default: none; set "all" for A100/H100/H200)
#   NSYS_CUDA_MEMORY      — Track CUDA memory (default: true)
#   NSYS_SAMPLE           — CPU sampling mode (default: none)
#   NSYS_RANKS_TO_PROFILE — Comma-separated ranks to profile, or "all" (default: all)
#   NSYS_BIN              — Explicit nsys binary path (auto-detected if unset)
#   NSYS_EXTRA_ARGS       — Additional nsys arguments
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
NSYS_DELAY="${NSYS_DELAY:-30}"
NSYS_DURATION="${NSYS_DURATION:-90}"
NSYS_OUTPUT_DIR="${NSYS_OUTPUT_DIR:-/local/nsight-reports}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt}"
NSYS_PYTORCH_MODE="${NSYS_PYTORCH_MODE:-autograd-shapes-nvtx}"
NSYS_PYTHON_SAMPLE="${NSYS_PYTHON_SAMPLE:-true}"
NSYS_GPU_METRICS="${NSYS_GPU_METRICS:-none}"
NSYS_CUDA_MEMORY="${NSYS_CUDA_MEMORY:-true}"
NSYS_SAMPLE="${NSYS_SAMPLE:-none}"
NSYS_RANKS_TO_PROFILE="${NSYS_RANKS_TO_PROFILE:-all}"
NSYS_BIN="${NSYS_BIN:-}"
NSYS_EXTRA_ARGS="${NSYS_EXTRA_ARGS:-}"

# ── Parse CLI arguments ──────────────────────────────────────────────────────
TRAINING_CMD=()
PARSING_OPTS=true

while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
        PARSING_OPTS=false
        shift
        continue
    fi
    if $PARSING_OPTS; then
        case "$1" in
            --delay=*) NSYS_DELAY="${1#*=}"; shift ;;
            --duration=*) NSYS_DURATION="${1#*=}"; shift ;;
            --output-dir=*) NSYS_OUTPUT_DIR="${1#*=}"; shift ;;
            --ranks=*) NSYS_RANKS_TO_PROFILE="${1#*=}"; shift ;;
            --help|-h)
                cat <<'USAGE'
nsys-profile.sh — Nsight Systems profiling wrapper for EKS distributed training

Usage: nsys-profile.sh [options] -- <training_command>

Options:
  --delay=N        Delay before collecting (seconds, default: 30)
  --duration=N     Collection duration (seconds, default: 90)
  --output-dir=DIR Directory for .nsys-rep files (default: /local/nsight-reports)
  --ranks=0,1|all  Ranks to profile (default: all)

Configuration: Set via environment variables (NSYS_DELAY, NSYS_DURATION, etc.)
See source header for full list.
USAGE
                exit 0 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    else
        TRAINING_CMD+=("$1")
        shift
    fi
done

if [[ ${#TRAINING_CMD[@]} -eq 0 ]]; then
    echo "ERROR: No training command specified."
    echo "Usage: nsys-profile.sh [options] -- <command>"
    exit 1
fi

# ── Auto-detect nsys binary ──────────────────────────────────────────────────
find_nsys() {
    if [[ -n "$NSYS_BIN" ]] && [[ -x "$NSYS_BIN" ]]; then
        echo "$NSYS_BIN"
        return
    fi
    # Search HyperPod, DLAMI, and mounted paths (newest first)
    for p in \
        /opt/nvidia/nsight-systems/*/bin/nsys \
        /opt/nvidia/nsight-systems/*/target-linux-x64/nsys \
        /nsight/*/bin/nsys \
        /nsight/*/target-linux-x64/nsys \
        /usr/local/cuda/nsight-systems-*/bin/nsys \
    ; do
        if [[ -x "$p" ]]; then
            echo "$p"
            return
        fi
    done
    if command -v nsys &>/dev/null; then
        command -v nsys
        return
    fi
    echo ""
}

NSYS_BIN=$(find_nsys)
if [[ -z "$NSYS_BIN" ]]; then
    echo "ERROR: nsys binary not found. Set NSYS_BIN or mount nsight-systems volume."
    echo "  HyperPod nodes: /opt/nvidia/nsight-systems/"
    echo "  Mount as hostPath and set NSYS_BIN, or mount at /nsight/"
    exit 1
fi

echo "=== Nsight Systems Profiler ==="
echo "nsys: $NSYS_BIN"
$NSYS_BIN --version
echo ""

# ── Check if this rank should be profiled ────────────────────────────────────
RANK="${RANK:-${SLURM_PROCID:-0}}"
LOCAL_RANK="${LOCAL_RANK:-${SLURM_LOCALID:-0}}"

should_profile() {
    [[ "$NSYS_RANKS_TO_PROFILE" == "all" ]] && return 0
    IFS=',' read -ra PROFILE_RANKS <<< "$NSYS_RANKS_TO_PROFILE"
    for r in "${PROFILE_RANKS[@]}"; do
        [[ "$RANK" == "$r" ]] && return 0
    done
    return 1
}

if ! should_profile; then
    echo "Rank $RANK not in profile list ($NSYS_RANKS_TO_PROFILE). Running without profiling."
    exec "${TRAINING_CMD[@]}"
fi

# ── Setup output ─────────────────────────────────────────────────────────────
mkdir -p "$NSYS_OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${NSYS_OUTPUT_DIR}/report_rank${RANK}_$(hostname)_${TIMESTAMP}"

# ── Build nsys command ───────────────────────────────────────────────────────
NSYS_CMD=(
    "$NSYS_BIN" profile
    --trace="$NSYS_TRACE"
    --sample="$NSYS_SAMPLE"
    --delay="$NSYS_DELAY"
    --duration="$NSYS_DURATION"
    --output="$OUTPUT_FILE"
    --force-overwrite=true
    --kill=none
    --stop-on-exit=true
)

# PyTorch autograd NVTX annotations (nsys >= 2024.5)
if [[ -n "$NSYS_PYTORCH_MODE" ]] && [[ "$NSYS_PYTORCH_MODE" != "none" ]]; then
    NSYS_CMD+=(--pytorch="$NSYS_PYTORCH_MODE")
fi

# Python call stack sampling
if [[ "$NSYS_PYTHON_SAMPLE" == "true" ]]; then
    NSYS_CMD+=(--python-sampling=true --python-sampling-frequency=1000)
fi

# GPU hardware metrics (not supported on A10G/g5 — use "none" for those)
if [[ -n "$NSYS_GPU_METRICS" ]] && [[ "$NSYS_GPU_METRICS" != "none" ]]; then
    NSYS_CMD+=(--gpu-metrics-devices="$NSYS_GPU_METRICS")
fi

# CUDA memory usage tracking
if [[ "$NSYS_CUDA_MEMORY" == "true" ]]; then
    NSYS_CMD+=(--cuda-memory-usage=true)
fi

# Auto-export stats and SQLite after collection
NSYS_CMD+=(--stats=true --export=sqlite)

# Extra arguments
if [[ -n "$NSYS_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    NSYS_CMD+=($NSYS_EXTRA_ARGS)
fi

# Append training command
NSYS_CMD+=("${TRAINING_CMD[@]}")

# ── Print config ─────────────────────────────────────────────────────────────
echo "Config:"
echo "  Rank:          $RANK (Local: $LOCAL_RANK)"
echo "  Delay:         ${NSYS_DELAY}s"
echo "  Duration:      ${NSYS_DURATION}s"
echo "  Output:        ${OUTPUT_FILE}.nsys-rep"
echo "  Traces:        $NSYS_TRACE"
echo "  PyTorch NVTX:  $NSYS_PYTORCH_MODE"
echo "  Python sample: $NSYS_PYTHON_SAMPLE"
echo "  GPU metrics:   $NSYS_GPU_METRICS"
echo "  CUDA memory:   $NSYS_CUDA_MEMORY"
echo ""
echo "Full command: ${NSYS_CMD[*]}"
echo "================================"
echo ""

exec "${NSYS_CMD[@]}"
