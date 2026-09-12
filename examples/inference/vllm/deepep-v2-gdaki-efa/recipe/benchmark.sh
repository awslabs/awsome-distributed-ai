#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# Concurrency sweep against a live serve. Writes fresh benchmarks/raw/<ts>/; exits non-zero on any failure.
set -euo pipefail
LEADER_IP="${1:?usage: benchmark.sh <leader-ip>}"
OUT_DIR="benchmarks/raw/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$OUT_DIR"
python3 "$(dirname "$0")/benchmark_probe.py" --url "http://${LEADER_IP}:8000/v1/chat/completions" \
  --out "${OUT_DIR}/sweep.jsonl" | tee "${OUT_DIR}/sweep.txt"
grep -q '"200"' "${OUT_DIR}/sweep.jsonl" || { echo "FAIL: no HTTP 200 in sweep — not a valid run"; exit 1; }
echo "results in ${OUT_DIR}"
