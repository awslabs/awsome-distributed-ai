# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Compare measured GEMM TFLOPS (results.json from run_gemm_bench.sh) against the DENSE
Blackwell roofline, and flag two failure modes:

  1. Reporting against the sparse peak by mistake (numbers look "too good").
  2. FP4 not actually engaged: NVFP4 measured at or below the MXFP8 number means the
     kernel silently upcast and is not using FP4 Tensor Cores.

DENSE peaks below are per-GB200 B200, approximate, and intentionally the DENSE figures
(not the 2x sparse marketing numbers). Override via env if your binning differs.
"""
import json
import os
import sys
from collections import defaultdict

# Dense per-GPU peak TFLOPS (approx, GB200 B200). Sparse would be ~2x these -- do not use.
PEAK = {
    "bf16": float(os.environ.get("PEAK_BF16", "2500")),
    "fp8": float(os.environ.get("PEAK_FP8", "5000")),
    "fp4": float(os.environ.get("PEAK_FP4", "10000")),
}


def family(recipe: str) -> str:
    if recipe.startswith("bf16"):
        return "bf16"
    if "fp4" in recipe or "nvfp4" in recipe:
        return "fp4"
    return "fp8"  # fp8_* and mxfp8


def main(argv):
    if len(argv) != 2:
        print("usage: roofline.py results.json", file=sys.stderr)
        return 2
    rows = json.load(open(argv[1]))

    by_shape = defaultdict(dict)
    for r in rows:
        key = (r["M"], r["K"], r["N"])
        if r.get("tflops") is not None:
            by_shape[key][r["recipe"]] = float(r["tflops"])

    ok = True
    for (M, K, N), recs in sorted(by_shape.items()):
        print(f"\n=== GEMM {M}x{K}x{N} ===")
        for recipe, tf in sorted(recs.items(), key=lambda kv: kv[1]):
            fam = family(recipe)
            peak = PEAK[fam]
            util = 100.0 * tf / peak if peak else 0.0
            note = ""
            if util > 100:
                note = "  <-- ABOVE dense peak: are you comparing to the sparse roofline?"
                ok = False
            print(f"  {recipe:14s} {tf:8.0f} TFLOPS  ({util:5.1f}% of dense {fam} peak {peak:.0f}){note}")

        # FP4-engaged check: NVFP4 should beat the best MXFP8 number.
        mxfp8 = max((v for k, v in recs.items() if "mxfp8" in k or k == "fp8_block"), default=None)
        nvfp4 = max((v for k, v in recs.items() if "fp4" in k), default=None)
        if mxfp8 and nvfp4:
            ratio = nvfp4 / mxfp8
            verdict = "OK" if ratio > 1.1 else "FP4 NOT ENGAGED (silent upcast?)"
            if ratio <= 1.1:
                ok = False
            print(f"  FP4/MXFP8 ratio = {ratio:.2f}  (expect ~1.46-1.66x) -> {verdict}")

    print("\nPASS" if ok else "\nFAIL (see notes above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
