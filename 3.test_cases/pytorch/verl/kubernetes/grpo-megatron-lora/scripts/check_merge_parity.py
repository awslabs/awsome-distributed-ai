# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Logit-parity gate for a merged Megatron+LoRA -> HF checkpoint.

Confirms the merged HF model produced by scripts/merge_megatron_lora_ckpt.py is
NOT a silent "base-only" merge (the historical failure mode) and is numerically sane.

It runs two cheap, single-GPU checks with plain HF transformers (no Megatron):

1. DIVERGENCE-FROM-BASE: the merged model's logits MUST differ from the base
   model's logits on the same prompts. If they are identical, the LoRA delta was
   lost -- the merge silently produced the base model. This is the primary gate.

2. SANITY: the merged model must produce finite logits (no NaN/Inf) and load
   cleanly via AutoModelForCausalLM.

This runs on ONE GPU with the two models loaded sequentially (not concurrently)
to keep memory bounded; for 235B use a CPU-offloaded / device_map="auto" load or
point --base-model / --merged-model at a small smoke checkpoint first.

Usage (small-model smoke -- recommended first):
    python scripts/check_merge_parity.py \
        --base-model /fsx/.../models/<small> \
        --merged-model /fsx/.../merged/<small>/step_N \
        --max-new-tokens 0 --num-prompts 4

For 235B, run as a Ray job on a GPU node (device_map="auto" across the 8 GPUs).
"""

from __future__ import annotations

import argparse
import sys

PROMPTS = [
    "def fibonacci(n):\n    ",
    "The capital of France is",
    "Write a Python function to reverse a string:\n",
    "Solve for x: 2x + 5 = 13.\n",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True, help="Base HF model dir")
    p.add_argument(
        "--merged-model",
        action="append",
        required=True,
        help="Merged HF model dir (output of the merger). Repeatable: pass --merged-model multiple "
        "times to check several checkpoints against ONE base load (much faster than re-running).",
    )
    p.add_argument("--num-prompts", type=int, default=4, help="How many of the built-in prompts to use")
    p.add_argument("--max-diff-tol", type=float, default=1e-3, help="Min mean abs logit diff to consider 'changed'")
    p.add_argument("--device-map", default="auto", help="HF device_map (auto for multi-GPU 235B; cuda:0 for small)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def _last_token_logits(model, tokenizer, prompts, device):
    import torch

    out = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**ids).logits[0, -1, :].float().cpu()
        out.append(logits)
    return out


def main() -> int:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("check_merge_parity").info

    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    prompts = PROMPTS[: args.num_prompts]
    merged_models = args.merged_model  # list (action=append)

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)

    # Load BASE once and cache its logits (reused across all merged checkpoints).
    log(f"Loading BASE model from {args.base_model} ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, device_map=args.device_map, trust_remote_code=args.trust_remote_code
    )
    base.eval()
    base_logits = _last_token_logits(base, tok, prompts, base.device)
    del base
    torch.cuda.empty_cache()
    log("Base logits cached; base model freed.")

    results = {}
    for merged_path in merged_models:
        log(f"--- Checking merged model: {merged_path} ---")
        merged = AutoModelForCausalLM.from_pretrained(
            merged_path, dtype=dtype, device_map=args.device_map, trust_remote_code=args.trust_remote_code
        )
        merged.eval()
        merged_logits = _last_token_logits(merged, tok, prompts, merged.device)
        del merged
        torch.cuda.empty_cache()

        # Sanity: finite logits
        finite = all(torch.isfinite(lg).all() for lg in merged_logits)
        # strict=True is load-bearing: a bare zip() truncates to the shorter list, so if either
        # _last_token_logits() call ever returns fewer rows than `prompts`, mean_diff would be
        # averaged over a SUBSET and this gate could still report PASS. This gate exists to be
        # trusted before spending eval GPU-hours, so a length mismatch must raise, not truncate.
        diffs = [(m - b).abs().mean().item() for m, b in zip(merged_logits, base_logits, strict=True)]
        mean_diff = sum(diffs) / len(diffs)
        passed = finite and mean_diff >= args.max_diff_tol
        results[merged_path] = (passed, mean_diff, finite)
        log(f"  finite={finite}  per-prompt={[round(d, 5) for d in diffs]}  mean|diff|={mean_diff:.6f}")
        log(f"  {'PASS' if passed else 'FAIL'}: {merged_path}")

    # Summary
    log("=" * 60)
    n_pass = sum(1 for p, _, _ in results.values() if p)
    for path, (passed, mean_diff, finite) in results.items():
        log(f"  [{'PASS' if passed else 'FAIL'}] mean|diff|={mean_diff:.4f} finite={finite}  {path}")
    log(f"{n_pass}/{len(results)} merged checkpoints PASSED parity (differ from base, finite logits).")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
