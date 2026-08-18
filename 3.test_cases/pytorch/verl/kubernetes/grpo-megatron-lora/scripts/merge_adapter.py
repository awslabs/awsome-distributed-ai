# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Merge a verl v0.8.0 native PEFT adapter into a standalone HuggingFace model.

Since verl v0.8.0, every Megatron+LoRA checkpoint save writes
a standard HF PEFT adapter to:

    <ckpt>/global_step_<N>/actor/huggingface/adapter/
        adapter_config.json      (peft_type=LORA, r, lora_alpha, target q/k/v/o_proj)
        adapter_model.safetensors

That is directly consumable by PEFT, so merging is now a plain:

    base = AutoModelForCausalLM.from_pretrained(<base>)
    merged = PeftModel.from_pretrained(base, <adapter>).merge_and_unload()
    merged.save_pretrained(<out>)

This REPLACES the old ~370-line Megatron-Bridge "replay" merger
(merge_megatron_lora_ckpt.py + ray_merge_launcher.py), which was only needed at
the pre-v0.8.0 pin where the stock verl merger could not reassemble Megatron
LoRA shards. See docs/eval-pipeline.md.

Safety gates:
  0. PROVENANCE (opt-in via --expect-rank/--expect-alpha, STRONGLY recommended):
     the adapter's rank/alpha must match what the caller intended. Gates 1 and 2
     and check_merge_parity.py are all provenance-BLIND -- they only ask "did
     something change?" -- so merging an adapter from the wrong experiment tree
     passes every one of them and ships the wrong model under the right name.
  1. NON-ZERO ADAPTER: every LoRA tensor in the adapter is checked to be non-zero
     before the merge, guarding against a silent base-only merge.
  2. DELTA-APPLIED: after merge_and_unload(), at least one merged weight must
     differ from the base weight it was derived from (sampled), proving the LoRA
     delta actually landed. (The full logit-parity gate is check_merge_parity.py.)

Runs on ONE GPU node. For 235B, load with device_map="auto" to shard the base
across the 8 GPUs of a single p6-b200 node. Submitted as a Ray job by
scripts/merge_adapter.sh.

Usage (standalone). Note --expect-rank: merge_adapter.sh defaults DATASET to
`mixed-code-math-v2`, which holds r=32 adapters, so an unguarded merge of an
r=128 run silently reads the wrong tree.
    CKPTS=/fsx/data/verl/ckpts/mixed-r128-cap32k-aug4b/megatron/Qwen3-235B-A22B
    python scripts/merge_adapter.py \
        --base-model   /fsx/data/verl/models/Qwen3-235B-A22B \
        --adapter-dir  "${CKPTS}"/global_step_100/actor/huggingface/adapter \
        --output-dir   /fsx/data/verl/merged/Qwen3-235B-A22B-mixed-r128-cap32k-aug4b/step_100 \
        --expect-rank 128 --expect-alpha 256
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base-model", required=True, help="Base HF model directory")
    p.add_argument(
        "--adapter-dir",
        required=True,
        help="PEFT adapter dir (…/actor/huggingface/adapter) with adapter_config.json + adapter_model.safetensors",
    )
    p.add_argument("--output-dir", required=True, help="Where to write the merged HF model")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Load/merge dtype (default bfloat16 — matches training)",
    )
    p.add_argument(
        "--device-map",
        default="auto",
        help='HF device_map ("auto" shards base across all visible GPUs; "cpu" for CPU-only)',
    )
    p.add_argument(
        "--skip-delta-check",
        action="store_true",
        help="Skip the post-merge base!=merged sanity check (not recommended)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --output-dir if it already exists",
    )
    p.add_argument(
        "--expect-rank",
        type=int,
        default=None,
        help=(
            "Assert the adapter's LoRA rank equals this before merging. Guards against "
            "merging an adapter from the wrong experiment tree -- every other gate here "
            "is provenance-blind and would report PASS. Strongly recommended."
        ),
    )
    p.add_argument(
        "--expect-alpha",
        type=int,
        default=None,
        help="Assert the adapter's lora_alpha equals this before merging",
    )
    return p.parse_args()


def _fail(msg: str) -> None:
    print(f"\n[merge_adapter] FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def _assert_adapter_provenance(cfg: dict, expect_rank, expect_alpha, adapter_dir: Path) -> None:
    """Gate 0: the adapter must be the one the caller intended.

    Every other gate in this pipeline is provenance-BLIND. `_assert_adapter_nonzero`,
    the delta gate, and check_merge_parity.py all only ask "did SOMETHING change?"
    -- so merging an adapter from a DIFFERENT experiment passes all three with a
    clean bill of health and ships the wrong model under the right name.

    That is not hypothetical here: merge_adapter.sh defaults DATASET to
    `mixed-code-math-v2`, which contains a full set of r=32/alpha=64 adapters
    including global_step_100. The r=128/alpha=256 run writes to a different tree.
    `merge_adapter.sh 100` with default arguments therefore silently merges the
    wrong experiment's r=32 adapter and every downstream check reports PASS.

    Cheapest possible gate: pure dict comparison, before any file is opened.
    """
    got_rank, got_alpha = cfg.get("r"), cfg.get("lora_alpha")
    if expect_rank is not None and got_rank != expect_rank:
        _fail(
            f"adapter RANK mismatch: expected r={expect_rank}, found r={got_rank} in "
            f"{adapter_dir}. This is almost certainly the wrong checkpoint tree -- "
            f"merge_adapter.sh defaults DATASET=mixed-code-math-v2 (r=32). Pass the "
            f"correct DATASET/CKPT_ROOT, or correct --expect-rank if this really is "
            f"the intended adapter."
        )
    if expect_alpha is not None and got_alpha != expect_alpha:
        _fail(
            f"adapter ALPHA mismatch: expected lora_alpha={expect_alpha}, found "
            f"{got_alpha} in {adapter_dir}. alpha/rank ratio is load-bearing for "
            f"comparability between runs -- refusing to merge."
        )
    if expect_rank is not None or expect_alpha is not None:
        print(
            f"[merge_adapter] provenance gate PASS: r={got_rank} alpha={got_alpha} "
            f"match expected (r={expect_rank}, alpha={expect_alpha})",
            flush=True,
        )
    else:
        print(
            "[merge_adapter] provenance gate SKIPPED: no --expect-rank/--expect-alpha "
            "given. Nothing downstream verifies WHICH adapter this is.",
            flush=True,
        )


def _assert_adapter_nonzero(adapter_dir: Path) -> int:
    """Gate 1: every adapter tensor must be non-zero. Returns tensor count."""
    from safetensors import safe_open

    st = adapter_dir / "adapter_model.safetensors"
    if not st.is_file():
        _fail(f"adapter_model.safetensors not found in {adapter_dir}")

    total = 0
    zero = []
    with safe_open(str(st), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            total += 1
            # count_nonzero avoids float underflow surprises of .any() on bf16
            if int(t.count_nonzero()) == 0:
                zero.append(k)
    if not keys:
        _fail("adapter safetensors contains no tensors")
    if zero:
        _fail(
            f"{len(zero)}/{total} adapter tensors are all-zero (silent base-only "
            f"adapter?). First few: {zero[:5]}"
        )
    print(f"[merge_adapter] adapter gate PASS: {total}/{total} tensors non-zero", flush=True)
    return total


def main() -> None:
    args = parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = Path(args.base_model)
    adapter = Path(args.adapter_dir)
    out = Path(args.output_dir)

    if not base.is_dir():
        _fail(f"--base-model not found: {base}")
    if not (adapter / "adapter_config.json").is_file():
        _fail(f"adapter_config.json not found in {adapter}")
    if out.exists():
        if args.overwrite:
            print(f"[merge_adapter] --overwrite: removing existing {out}", flush=True)
            shutil.rmtree(out)
        elif any(out.iterdir()):
            _fail(f"--output-dir exists and is non-empty: {out} (use --overwrite)")
    out.mkdir(parents=True, exist_ok=True)

    # Log the adapter config for the record.
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    print(
        f"[merge_adapter] adapter: r={cfg.get('r')} alpha={cfg.get('lora_alpha')} "
        f"targets={cfg.get('target_modules')} base_ref={cfg.get('base_model_name_or_path')}",
        flush=True,
    )

    # Gate 0 — provenance: is this the adapter the caller MEANT? Must run before
    # every other gate, all of which are provenance-blind.
    _assert_adapter_provenance(cfg, args.expect_rank, args.expect_alpha, adapter)

    # Gate 1 — non-zero adapter (cheap, CPU, before we spend minutes loading 235B).
    _assert_adapter_nonzero(adapter)

    dtype = getattr(torch, args.dtype)

    t0 = time.time()
    print(f"[merge_adapter] loading base model from {base} (device_map={args.device_map})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base),
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"[merge_adapter] base loaded in {time.time() - t0:.0f}s", flush=True)

    # Sample a few base weights BEFORE merge for the delta check (Gate 2).
    sampled = {}
    if not args.skip_delta_check:
        for name, p in model.named_parameters():
            # q/k/v/o_proj are the adapter targets — pick ones likely to change.
            if any(t in name for t in ("q_proj", "k_proj", "v_proj", "o_proj")) and name.endswith(".weight"):
                sampled[name] = p.detach().float().flatten()[:4096].clone().cpu()
            if len(sampled) >= 6:
                break

    t1 = time.time()
    print(f"[merge_adapter] applying PEFT adapter from {adapter}...", flush=True)
    peft_model = PeftModel.from_pretrained(model, str(adapter), torch_dtype=dtype)
    print("[merge_adapter] merge_and_unload()...", flush=True)
    merged = peft_model.merge_and_unload()
    print(f"[merge_adapter] merged in {time.time() - t1:.0f}s", flush=True)

    # Gate 2 — at least one sampled weight must have changed.
    if not args.skip_delta_check and sampled:
        merged_params = dict(merged.named_parameters())
        changed = 0
        max_delta = 0.0
        for name, before in sampled.items():
            after = merged_params[name].detach().float().flatten()[:4096].cpu()
            d = float((after - before).abs().max())
            max_delta = max(max_delta, d)
            if d > 1e-6:
                changed += 1
        if changed == 0:
            _fail(
                "delta gate: NONE of the sampled q/k/v/o_proj weights changed after "
                "merge — the LoRA delta was not applied (silent base-only merge)."
            )
        print(
            f"[merge_adapter] delta gate PASS: {changed}/{len(sampled)} sampled "
            f"weights changed (max |Δ|={max_delta:.4g})",
            flush=True,
        )

    t2 = time.time()
    print(f"[merge_adapter] saving merged model to {out}...", flush=True)
    merged.save_pretrained(str(out), safe_serialization=True)

    # Tokenizer + configs: prefer the checkpoint's huggingface/ dir (has the exact
    # tokenizer used in training); fall back to base.
    tok_src = adapter.parent  # …/actor/huggingface (tokenizer lives one level up)
    if not (tok_src / "tokenizer_config.json").is_file():
        tok_src = base
    print(f"[merge_adapter] copying tokenizer/config from {tok_src}", flush=True)
    tok = AutoTokenizer.from_pretrained(str(tok_src), trust_remote_code=True)
    tok.save_pretrained(str(out))

    print(f"[merge_adapter] save done in {time.time() - t2:.0f}s", flush=True)
    print(
        f"\n[merge_adapter] SUCCESS — merged model at {out} "
        f"(total {time.time() - t0:.0f}s)",
        flush=True,
    )
    print(
        "[merge_adapter] NEXT: run scripts/check_merge_parity.py for the full "
        "logit-parity gate before serving.",
        flush=True,
    )


if __name__ == "__main__":
    main()
