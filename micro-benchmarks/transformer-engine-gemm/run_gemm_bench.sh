#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Sweep GEMM throughput across precisions on GB200 using Transformer Engine's
# benchmark_gemm.py, and write results.json for roofline.py.
#
# Recipes: BF16 baseline, FP8 (delayed/current/block), MXFP8, NVFP4.
# Shapes: large square + a couple of LLM-shaped GEMMs where FP4 helps most.

set -euo pipefail

: "${SHAPES:=8192x8192x8192 16384x16384x16384 4096x4096x28672 8192x8192x28672}"
: "${RECIPES:=bf16 fp8_delayed fp8_current fp8_block mxfp8 nvfp4}"
: "${OUT:=results.json}"

# Locate TE's benchmark script from the installed package.
BENCH="$(python3 - <<'PY'
import os, transformer_engine as te
root = os.path.dirname(os.path.dirname(te.__file__))
for cand in (
    os.path.join(root, "transformer_engine", "benchmarks", "gemm", "benchmark_gemm.py"),
    os.path.join(root, "benchmarks", "gemm", "benchmark_gemm.py"),
):
    if os.path.exists(cand):
        print(cand); break
PY
)"

if [[ -z "${BENCH:-}" || ! -f "$BENCH" ]]; then
  echo "ERROR: benchmark_gemm.py not found in the installed Transformer Engine." >&2
  echo "Check the TE version (need >= 2.16) and that the benchmarks/ dir ships in this build." >&2
  exit 1
fi

echo "Using TE benchmark: $BENCH"
echo "Shapes:  $SHAPES"
echo "Recipes: $RECIPES"

# benchmark_gemm.py flags vary slightly across TE minor versions; this drives the
# documented --recipe / --shape interface. Adjust flag names to your TE if it differs.
tmp="$(mktemp)"
echo "[]" > "$OUT"
for shape in $SHAPES; do
  IFS='x' read -r M K N <<< "$shape"
  for r in $RECIPES; do
    echo "=== shape ${M}x${K}x${N}  recipe ${r} ==="
    python3 "$BENCH" --m "$M" --k "$K" --n "$N" --recipe "$r" --json "$tmp" || {
      echo "  (recipe $r failed on this TE/shape -- recorded as null)"; echo "{}" > "$tmp"; }
    python3 - "$OUT" "$tmp" "$M" "$K" "$N" "$r" <<'PY'
import json, sys
out, tmp, M, K, N, r = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
data = json.load(open(out))
try:
    rec = json.load(open(tmp))
except Exception:
    rec = {}
data.append({"M": M, "K": K, "N": N, "recipe": r,
             "tflops": rec.get("tflops"), "time_ms": rec.get("time_ms")})
json.dump(data, open(out, "w"), indent=2)
PY
  done
done
echo "Wrote $OUT"
