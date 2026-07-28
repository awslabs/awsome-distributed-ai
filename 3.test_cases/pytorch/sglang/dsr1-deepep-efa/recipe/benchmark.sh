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
# FIXED SEQUENCE LENGTHS. --random-range-ratio defaults to 0.0, which samples every
# length uniformly from [0, len] instead of using len: a "256 in / 512 out, conc 64"
# point then pushes ~half the tokens, and short requests keep draining the batch so
# sustained concurrency lands near 46, not 64. Measured on 2 x p5en, same live server,
# only this flag differing: 374 tok/s at ratio 0.0 vs 1127 at ratio 1.0, with mean E2E
# latency ~equal. So we pin 1.0. RANDOM_RANGE_RATIO overrides it if you want variable
# lengths.
#
# WARMUP. --warmup-requests defaults to 1, which only exercises the batch-1 path. We
# scale it to each point's concurrency so the server reaches the batch shape about to
# be measured. WARMUP_REQUESTS=0 disables it.

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
    # Mount the HF cache: --dataset-name random still downloads the ShareGPT blob
    # (256 MB) to source its token ids, and the container is --rm, so an unmounted
    # cache refetches it per point -- and a stalled fetch looks exactly like a hung
    # benchmark (idle GPUs, no output) against a healthy server.
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
