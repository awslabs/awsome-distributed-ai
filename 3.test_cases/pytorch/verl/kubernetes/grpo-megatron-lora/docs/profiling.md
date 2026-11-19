<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->
# Profiling

Built-in metrics, the PyTorch profiler, Nsight Systems, the memory profiler, and how to
view traces in Perfetto.

> **Placeholders.** `${S3_BUCKET_NAME}` is defined by `env_vars.example`; run
> `source env_vars` first.

verl includes a built-in 3-layer profiling system. This project exposes it through a `profiling` config group with presets.

## Built-in Metrics (Always Active)

Every training run logs `timing_s/*` (per-step wall clock) and `perf/throughput` metrics automatically — no profiling config needed.

## PyTorch Profiler

```bash
python3 scripts/submit_training.py profiling=torch
```

Profiles actor rank 0 on steps 3-5. View traces in [Perfetto](https://ui.perfetto.dev/) or `chrome://tracing`.

## Nsight Systems

```bash
python3 scripts/submit_training.py profiling=nsys
```

Deep GPU kernel and NCCL analysis on all ranks (step 5 only). View reports with `nsys-ui`.

## Memory Profiler

```bash
python3 scripts/submit_training.py profiling=memory
```

CUDA memory snapshots for OOM/fragmentation debugging. View with [PyTorch Memory Viz](https://pytorch.org/memory_viz).

## Custom Overrides

```bash
# Profile different steps and ranks
python3 scripts/submit_training.py profiling=torch profiling.steps='[2,3]' profiling.actor_ranks='[0,1]'

# Enable critic profiling alongside actor
python3 scripts/submit_training.py profiling=torch profiling.critic_enable=true
```

## Output Location

Traces are saved to `/fsx/data/verl/profiling/{model_name}/` on FSx and synced to S3
via the FSx Data Repository Association at
`s3://${S3_BUCKET_NAME}/profiling/`.

**Caveat**: Pod replacement loses `/tmp` data — always use FSx paths (the default). Nsight Systems with async rollout may require `discrete: True` on the rollout config.

## Viewing Traces with Perfetto

[Perfetto](https://ui.perfetto.dev/) is the recommended viewer for PyTorch profiler
traces. It runs in the browser with no installation required and handles 100 MB+ traces
well.

**1. Download traces from S3:**

```bash
# List available profiles
aws s3 ls s3://${S3_BUCKET_NAME}/profiling/ --recursive --human-readable

# Download all traces for a model
aws s3 cp s3://${S3_BUCKET_NAME}/profiling/Qwen3-235B-A22B/ \
  ./profiling/ --recursive

# Or download a single trace
aws s3 cp s3://${S3_BUCKET_NAME}/profiling/Qwen3-235B-A22B/prof_rank-0_*.json.gz \
  ./profiling/
```

**2. Open in Perfetto:**

1. Go to https://ui.perfetto.dev/
2. Click **"Open trace file"** or drag and drop a `.json.gz` file
3. Perfetto handles gzipped traces natively — no need to decompress

**3. Navigation:**

| Key | Action |
|-----|--------|
| `W` / `S` | Zoom in / out |
| `A` / `D` | Pan left / right |
| Scroll wheel | Zoom at cursor |
| Click a slice | View kernel details (duration, arguments) |

**What to look for:**

- **CUDA stream rows** — which GPU kernels dominate (MoE all-to-all, attention, LoRA)
- **CPU→GPU gaps** — long gaps between kernel launches indicate Python/Ray scheduling overhead
- **Recompilation** — if step 3 (first profiled) is much slower than step 5, JIT warmup is the cause
- **Memory events** — allocation/free patterns if memory profiling was enabled
