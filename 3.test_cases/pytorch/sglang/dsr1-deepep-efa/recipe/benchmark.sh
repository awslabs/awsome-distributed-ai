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

set -euo pipefail

STAGE=${1:?need STAGE: prefill | decode}
: "${IMAGE_URI:?source setup/env_vars first}"
: "${MODEL:?source setup/env_vars first}"
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

    # Run the bench client inside the image so the harness version matches the
    # server's. --network host so localhost reaches the server on this node.
    docker run --rm --network host \
        -v "${OUT_DIR}:/out" \
        --entrypoint python3 "$IMAGE_URI" \
        -m sglang.bench_serving \
            --backend sglang \
            --model "${MODEL}" \
            --host 127.0.0.1 --port "${SERVE_PORT}" \
            --dataset-name random \
            --random-input-len "${in_len}" \
            --random-output-len "${out_len}" \
            --max-concurrency "${conc}" \
            --num-prompts "${num_prompts}" \
            --output-file "/out/${tag}.jsonl" \
        || echo "WARN: ${tag} failed; continuing"
done

echo
echo "==> Done. Raw results in ${OUT_DIR}"
echo "Repeat with MOE_BACKEND=baseline (and =tp) after restarting the server to compare."
