#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Benchmark a running server from NODE 0. Prefill and decode are swept
# separately, following standard LLM-serving practice, because the two stages
# stress the MoE all-to-all very differently:
#
#   prefill : --random-output-len 1  => TTFT is ~pure prefill time
#   decode  : short input, long output => TPOT-dominated
#
# Usage:
#   source setup/env_vars
#   recipe/benchmark.sh prefill [OUT_DIR]
#   recipe/benchmark.sh decode  [OUT_DIR]
#
# Results land as one JSON per point in OUT_DIR (default: benchmarks/raw/$MOE_BACKEND).
# Re-run each config across several seeds before quoting numbers anywhere: a
# single pass is a smoke benchmark, not a tight measurement.
#
# FIXED SEQUENCE LENGTHS. --random-range-ratio must be set explicitly. It defaults
# to 0.0, which samples every length uniformly from [0, len] rather than using len:
# a "256 in / 512 out, conc 64" point then actually pushes ~half the tokens, and
# short requests keep draining the batch so sustained concurrency lands near 46, not
# 64. Measured on 2 x p5en (H200), MOE_BACKEND=deepep, same live server, the same
# nominal 256/512/conc-64 point, differing only in this flag:
#
#   ratio 0.0 (default): 16225 in / 33252 out tok, concurrency 46.4,  374 tok/s
#   ratio 1.0 (fixed)  : 32768 in / 65536 out tok, concurrency 63.9, 1127 tok/s
#
# Mean E2E latency is ~equal in both (32.2 s vs 29.1 s) -- the 3x tok/s difference is
# the workload changing, not the server getting faster. So the default makes tok/s
# incomparable across any two runs that did not use identical seeds, and makes the
# `conc` in a filename overstate the concurrency actually sustained. Fixed lengths
# are also what the archived tables on this page were measured with.
#
# RANDOM_RANGE_RATIO can override it if you specifically want variable lengths.
#
# WARMUP. bench_serving's --warmup-requests defaults to 1, which only exercises the
# batch-1 code path. We scale it to each point's concurrency so the server reaches
# the batch shape about to be measured before measurement starts. On this image the
# DeepGEMM kernels are all JIT-compiled during server startup (before the health
# endpoint comes up), so this is cheap insurance against first-request effects --
# scheduler ramp, CUDA graph selection -- not a fix for compile cost in the
# measurement. WARMUP_REQUESTS=0 disables it.

set -euo pipefail

STAGE=${1:?need STAGE: prefill | decode}
: "${IMAGE_URI:?source setup/env_vars first}"
: "${MODEL:?source setup/env_vars first}"
: "${HF_CACHE_DIR:?source setup/env_vars first}"
MOE_BACKEND=${MOE_BACKEND:-deepep}
SERVE_PORT=${SERVE_PORT:-30000}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR=${2:-"${PROJECT_DIR}/benchmarks/raw/${MOE_BACKEND}"}
mkdir -p "$OUT_DIR"

if ! curl -sf "localhost:${SERVE_PORT}/health" >/dev/null; then
    echo "ERROR: no healthy server on localhost:${SERVE_PORT}." >&2
    echo "Start it with recipe/serve.sh on both nodes and wait for load to finish." >&2
    exit 1
fi

# in_len:out_len:concurrency:num_prompts
case "$STAGE" in
    prefill)
        POINTS=(1024:1:1:16 4096:1:1:16 8192:1:1:16 4096:1:8:64 4096:1:32:128) ;;
    decode)
        POINTS=(256:512:32:64 256:512:64:128 256:512:256:512 256:512:512:1024) ;;
    *)
        echo "ERROR: STAGE must be prefill or decode" >&2; exit 1 ;;
esac

echo "==> ${STAGE} sweep, backend=${MOE_BACKEND}, ${#POINTS[@]} points -> ${OUT_DIR}"

for point in "${POINTS[@]}"; do
    IFS=: read -r in_len out_len conc num_prompts <<< "$point"
    tag="${STAGE}_in${in_len}_out${out_len}_conc${conc}"
    echo
    echo "--- ${tag} ---"

    # Warm at this point's concurrency (bench_serving caps warmup output_len at 32).
    warmup=${WARMUP_REQUESTS:-$conc}

    # Run the bench client inside the image so the harness version matches the
    # server's. --network host so localhost reaches the server on this node.
    #
    # The HF cache mount is not optional. --dataset-name random still calls
    # download_and_cache_hf_file() for the ShareGPT blob (256 MB) to source its
    # token ids, and the container is --rm, so an unmounted cache re-downloads it
    # once per point. Worse, it can wedge: measured 2026-07-28 on p5en, the client
    # sat 20+ min at 0% CPU inside huggingface_hub's xet_get with the download
    # stalled, which looks exactly like a hung benchmark (idle GPUs, no output)
    # while the server is perfectly healthy. Sharing the server's cache makes the
    # blob a one-time cost, and HF_HUB_OFFLINE=0 is left alone so the first fetch
    # still works.
    docker run --rm --network host \
        -v "${OUT_DIR}:/out" \
        -v "${HF_CACHE_DIR}:/hf" -e HF_HOME=/hf \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        --entrypoint python3 "$IMAGE_URI" \
        -m sglang.bench_serving \
            --backend sglang \
            --model "${MODEL}" \
            --host 127.0.0.1 --port "${SERVE_PORT}" \
            --dataset-name random \
            --random-input-len "${in_len}" \
            --random-output-len "${out_len}" \
            --random-range-ratio "${RANDOM_RANGE_RATIO:-1.0}" \
            --seed "${SEED:-42}" \
            --max-concurrency "${conc}" \
            --num-prompts "${num_prompts}" \
            --warmup-requests "${warmup}" \
            --output-file "/out/${tag}.jsonl" \
        || echo "WARN: ${tag} failed; continuing"
done

echo
echo "==> Done. Raw results in ${OUT_DIR}"
echo "Repeat with MOE_BACKEND=baseline (and =tp) after restarting the server to compare."
