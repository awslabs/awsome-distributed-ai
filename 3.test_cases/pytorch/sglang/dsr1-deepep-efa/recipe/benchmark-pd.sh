#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Benchmark a running 2P2D deployment through the router. Prefill and decode are
# swept separately because the two stages stress the MoE all-to-all in opposite
# directions (see benchmarks/README.md, where the two stages have
# different winners):
#
#   prefill : --random-output-len 1     => TTFT is ~pure prefill time
#   decode  : short input, long output  => TPOT-dominated
#
# Usage:
#   source setup/env_vars
#   recipe/benchmark-pd.sh decode  [OUT_DIR]
#   recipe/benchmark-pd.sh prefill [OUT_DIR]
#
# Run decode BEFORE prefill. The prefill sweep's high-concurrency points push
# the 2-node prefill role hard enough to occasionally destabilise it, and there
# is no reason to risk the decode data to find that out.
#
# Results land as one JSON per point in OUT_DIR
# (default: benchmarks/raw/2p2d/$MOE_BACKEND[-dp]).
# A single pass is a smoke benchmark; repeat across seeds before quoting numbers.

set -euo pipefail

STAGE=${1:?need STAGE: prefill | decode}
: "${IMAGE_URI:?source setup/env_vars first}"
: "${MODEL:?source setup/env_vars first}"
: "${ROUTER_IP:?source setup/env_vars first}"
: "${HF_CACHE_DIR:?source setup/env_vars first}"
MOE_BACKEND=${MOE_BACKEND:-deepep}
DP_ATTENTION=${DP_ATTENTION:-0}
ROUTER_PORT=${ROUTER_PORT:-8000}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SUFFIX=""
[[ "$DP_ATTENTION" == "1" ]] && SUFFIX="-dp"
OUT_DIR=${2:-"${PROJECT_DIR}/benchmarks/raw/2p2d/${MOE_BACKEND}${SUFFIX}"}
mkdir -p "$OUT_DIR"

BASE_URL="http://${ROUTER_IP}:${ROUTER_PORT}"
if ! curl -sf "${BASE_URL}/health" >/dev/null; then
    echo "ERROR: no healthy router at ${BASE_URL}." >&2
    echo "Start all four servers with recipe/serve-pd.sh serve, then 'router'." >&2
    exit 1
fi

# in_len:out_len:concurrency:num_prompts
case "$STAGE" in
    prefill)
        # Concurrency is swept per input length and tapers as inputs grow: a
        # single 64K request already fills the prefill batch, and 64K at
        # concurrency >= 4 exhausts the 2-node prefill KV pool.
        #
        # 128K is deliberately absent. Prefill computes fine, but the Mooncake
        # transfer of a 128K-token KV cache times out with
        #   EFA submitSlicesOnPeer: CQ drain wr_depth=256, max=256
        # so the 2P2D input ceiling here is a KV-transfer limit, not a compute
        # one. Raising MC_MAX_WR or chunking the KV transfer is the thing to try.
        POINTS=(
            1024:1:1:8      1024:1:8:64     1024:1:64:192   1024:1:256:512  1024:1:512:1024
            4096:1:1:8      4096:1:8:32     4096:1:32:128   4096:1:128:256
            16384:1:1:4     16384:1:4:16    16384:1:16:64   16384:1:32:128
            32768:1:1:4     32768:1:4:16    32768:1:8:32
            65536:1:1:2     65536:1:2:4
        ) ;;
    decode)
        # A: long-output scaling at fixed concurrency 32 (2048 drops to 24 to
        #    bound runtime). B: concurrency ramp at output 512.
        POINTS=(
            256:256:32:64   256:512:32:64   256:1024:32:64  256:2048:24:48
            256:512:16:32   256:512:64:128  256:512:96:192  256:512:128:256
        ) ;;
    *)
        echo "ERROR: STAGE must be prefill or decode" >&2; exit 1 ;;
esac

echo "==> ${STAGE} sweep, backend=${MOE_BACKEND}${SUFFIX}, ${#POINTS[@]} points -> ${OUT_DIR}"

for point in "${POINTS[@]}"; do
    IFS=: read -r in_len out_len conc num_prompts <<< "$point"
    tag="${STAGE}_in${in_len}_out${out_len}_conc${conc}"
    echo
    echo "--- ${tag} ---"

    # --pd-separated makes the harness attribute TTFT/TPOT correctly across the
    # disaggregated roles; --random-range-ratio 1 pins the input length exactly
    # (the default jitters it, which blurs the per-length comparison).
    # Mount the HF cache -- see the note in benchmark.sh.
    docker run --rm --network host --privileged \
        -v "${OUT_DIR}:/out" \
        -v "${HF_CACHE_DIR}:/hf" -e HF_HOME=/hf \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        --entrypoint python3 "$IMAGE_URI" \
        -m sglang.bench_serving \
            --backend sglang-oai \
            --base-url "${BASE_URL}" \
            --model "${MODEL}" \
            --pd-separated \
            --dataset-name random \
            --random-input-len "${in_len}" \
            --random-output-len "${out_len}" \
            --random-range-ratio 1 \
            --max-concurrency "${conc}" \
            --num-prompts "${num_prompts}" \
            --output-file "/out/${tag}.jsonl" \
        || echo "WARN: ${tag} failed; continuing"
done

echo
echo "==> Done. Raw results in ${OUT_DIR}"
echo "Compare by re-running with MOE_BACKEND=baseline, =tp, and DP_ATTENTION=1"
echo "after relaunching all four servers."
