#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Warm up and run a rate sweep against a deployed topology, repeating each
# rate once per seed so users can aggregate over multiple runs.
#
# Usage:
#   ./recipe/benchmark.sh <topology> [results_dir]
#
# Where <topology> is one of unified|1p1d|1p3d|1p4d|1p5d. The script targets
# http://${PREFILL_IP}:${PORT} where PORT=8000 (proxy for disaggregated, vLLM
# directly for unified).
#
# Override which seeds and rates run via environment variables:
#   SEEDS="1234 5678 9012"   # default — 3 seeds
#   RATES="1 4 8 16 inf"     # default
#   NUM_PROMPTS=50           # default

set -euo pipefail

TOPOLOGY="${1:-}"
RESULT_DIR="${2:-results/${TOPOLOGY}}"

if [[ -z "${TOPOLOGY}" ]]; then
    echo "Usage: $0 <unified|1p1d|1p3d|1p4d|1p5d> [results_dir]" >&2
    exit 1
fi
if [[ -z "${PREFILL_IP:-}" || -z "${MODEL:-}" ]]; then
    echo "ERROR: source setup/env_vars first." >&2
    exit 1
fi

SEEDS="${SEEDS:-1234 5678 9012}"
RATES="${RATES:-1 4 8 16 inf}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"

ENDPOINT="http://${PREFILL_IP}:8000"

mkdir -p "${RESULT_DIR}"

echo "==> Warmup against ${ENDPOINT}"
vllm bench serve \
    --base-url "${ENDPOINT}" \
    --model "${MODEL}" \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 256 \
    --num-prompts 10 \
    --request-rate 1 \
    --seed 1234

for SEED in ${SEEDS}; do
    SEED_DIR="${RESULT_DIR}/seed-${SEED}"
    mkdir -p "${SEED_DIR}"
    echo "==> Rate sweep seed=${SEED} (results -> ${SEED_DIR})"
    for RATE in ${RATES}; do
        echo "----- seed=${SEED} rate=${RATE} -----"
        vllm bench serve \
            --base-url "${ENDPOINT}" \
            --model "${MODEL}" \
            --dataset-name random \
            --random-input-len 1024 \
            --random-output-len 256 \
            --num-prompts "${NUM_PROMPTS}" \
            --request-rate "${RATE}" \
            --metric-percentiles "50,90,95,99" \
            --seed "${SEED}" \
            --save-result \
            --result-dir "${SEED_DIR}"
    done
done

echo "==> Done. Result JSON files in ${RESULT_DIR}/seed-*/"
