#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Sweep the 2P2D deployment through the router. Prefill and decode are swept
# separately, because the two stages stress the MoE all-to-all very differently:
# prefill moves large token batches through the dispatch/combine kernels once,
# while decode pays the per-layer launch + RDMA cost on every single token.
#
# Usage:
#   source setup/env_vars
#   recipe/benchmark.sh decode         # short in, fixed out => TPOT-dominated
#   recipe/benchmark.sh prefill        # out=1 => TTFT is ~pure prefill
#
# Raw JSON lands in benchmarks/raw/$MOE_BACKEND/.
#
# BENCH_TOOL selects the load generator:
#   vllm   (default) `vllm bench serve`, already in this image.
#   sglang            `sglang.bench_serving --pd-separated`, which needs the
#                     sglang image (SGLANG_IMAGE) but is what produced the
#                     numbers in benchmarks/RESULTS.md — see the note there.
#
# ORDER MATTERS: run `decode` BEFORE `prefill` on a freshly started pair of
# engines. The prefill sweep's larger batches can destabilise a 2-node TP16
# prefill role, and a wedged prefill role costs you the decode numbers too.

set -euo pipefail

KIND=${1:?need prefill | decode}

: "${IMAGE_URI:?source setup/env_vars first}"
MOE_BACKEND=${MOE_BACKEND:-deepep}
ROUTER_HOST=${ROUTER_HOST:-127.0.0.1}
ROUTER_PORT=${ROUTER_PORT:-8000}
ROUTER="http://${ROUTER_HOST}:${ROUTER_PORT}"
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-R1}
BENCH_TOOL=${BENCH_TOOL:-vllm}
SGLANG_IMAGE=${SGLANG_IMAGE:-lmsysorg/sglang:v0.5.13.post1-cu130}
# Per-point wall-clock cap. PD-disaggregated decode occasionally wedges a request
# under concurrency; without a cap one bad point hangs the whole sweep.
CAP=${CAP:-900}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/benchmarks/raw/${MOE_BACKEND}"
mkdir -p "$OUT_DIR"

bench() {  # <input_len> <output_len> <concurrency> <num_prompts>
    local il=$1 ol=$2 conc=$3 np=$4
    local label="${KIND}-in${il}-out${ol}-c${conc}"
    local cname="bench-${MOE_BACKEND}-${il}-${ol}-${conc}"
    echo "=== [${MOE_BACKEND}/${BENCH_TOOL}] in=${il} out=${ol} conc=${conc} prompts=${np} ==="

    if [[ "$BENCH_TOOL" == "sglang" ]]; then
        # NO --pd-separated here. That flag speaks sglang_router's PD protocol;
        # vllm-router does not implement it, so it must be driven as a plain
        # OpenAI-compatible endpoint. (The sglang sample in this repo DOES need
        # --pd-separated — do not copy that invocation over.)
        timeout --kill-after=30 "$CAP" \
        docker run --rm --name "$cname" --network host \
            --entrypoint bash "$SGLANG_IMAGE" -c "
                python3 -m sglang.bench_serving \
                    --backend sglang-oai --base-url ${ROUTER} \
                    --model ${SERVED_MODEL_NAME} \
                    --dataset-name random \
                    --random-input-len ${il} --random-output-len ${ol} \
                    --num-prompts ${np} --max-concurrency ${conc} \
                    --random-range-ratio 1
            " 2>&1 | tee "${OUT_DIR}/${label}.log" \
              | grep -iE "Successful requests|Input token throughput|Output token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Request throughput" || true
    else
        # --ignore-eos so every request generates exactly output_len tokens;
        # otherwise R1 stops early and the decode points are not comparable.
        timeout --kill-after=30 "$CAP" \
        docker run --rm --name "$cname" --network host \
            -v "${OUT_DIR}:/results" \
            --entrypoint bash "$IMAGE_URI" -c "
                vllm bench serve \
                    --backend openai-chat --endpoint /v1/chat/completions \
                    --base-url ${ROUTER} \
                    --model ${SERVED_MODEL_NAME} \
                    --dataset-name random \
                    --random-input-len ${il} --random-output-len ${ol} \
                    --random-range-ratio 1 \
                    --num-prompts ${np} --max-concurrency ${conc} \
                    --ignore-eos \
                    --percentile-metrics ttft,tpot,itl,e2el \
                    --save-result --result-dir /results \
                    --result-filename ${label}.json
            " 2>&1 | tee "${OUT_DIR}/${label}.log" \
              | grep -iE "Successful requests|Total Token throughput|Output token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Request throughput" || true
    fi

    docker rm -f "$cname" >/dev/null 2>&1 || true
    echo "---"
}

case "$KIND" in
    decode)
        # output_len 128, not 512: at TP16 2-node PD-disagg a 512-token generation
        # pushes a point past the ${CAP}s cap. 128 tokens is still a real
        # multi-step decode workload. Concurrency stops at 32 — higher floods the
        # 2-node prefill role into a watchdog self-kill.
        for C in 8 16 32; do bench 256 128 "$C" $((C * 4)); done
        ;;
    prefill)
        # output_len 1, so TTFT is essentially prefill time. Single-request points
        # plus one light conc=4 point for batched throughput: conc >= 8 at 4096
        # tokens crashes a 2-node TP16 prefill role on H100 80GB.
        bench 1024 1 1 4
        bench 4096 1 1 4
        bench 8192 1 1 4
        bench 4096 1 4 16
        ;;
    *)
        echo "ERROR: argument must be prefill or decode" >&2; exit 1 ;;
esac

echo "=== ${KIND} sweep done (backend=${MOE_BACKEND}) -> ${OUT_DIR} ==="
