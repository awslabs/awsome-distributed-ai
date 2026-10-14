#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# Concurrency sweep against a live serve. Writes fresh benchmarks/raw/<ts>/; exits non-zero on any failure.
# The pass/fail authority is benchmark_probe.py's exit code (0 only if EVERY request at
# EVERY level succeeded) — no separate grep gate that could disagree with it.
set -euo pipefail
LEADER_IP="${1:?usage: benchmark.sh <leader-ip>}"
# OUT_ROOT: benchmarks/raw in the repo checkout; inside the pod set OUT_ROOT=/work/benchmarks
OUT_DIR="${OUT_ROOT:-benchmarks/raw}/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$OUT_DIR"
rc=0
python3 "$(dirname "$0")/benchmark_probe.py" --url "http://${LEADER_IP}:8000/v1/chat/completions" \
  --out "${OUT_DIR}/sweep.jsonl" | tee "${OUT_DIR}/sweep.txt" || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: probe reported failed requests (rc=$rc) — not a valid run"; exit "$rc"; }
echo "results in ${OUT_DIR}"
