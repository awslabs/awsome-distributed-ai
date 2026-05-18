#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Before/after eval for Nemotron-Mini-4B on Goldilocks math problems.

Evaluates base model and optionally the GRPO-trained LoRA checkpoint
on 200 held-out math problems with Python-verified answers.

Usage:
  # Base model only:
  python3 eval_nemotron_goldilocks.py --base-only --dataset /path/to/train.jsonl

  # Compare base vs trained:
  python3 eval_nemotron_goldilocks.py \\
    --model nvidia/Nemotron-Mini-4B-Instruct \\
    --dataset /path/to/goldilocks/train.jsonl \\
    --checkpoint-dir /path/to/checkpoints \\
    --output results.json

All paths are configurable via CLI arguments. Defaults use SHARED_DIR env var
(fallback: /shared/nvrx-demo) as the base directory.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

_SHARED_DIR = os.environ.get("SHARED_DIR", "/shared/nvrx-demo")
os.environ.setdefault("HF_HOME", os.path.join(_SHARED_DIR, "hf_cache"))

DEFAULT_MODEL = "nvidia/Nemotron-Mini-4B-Instruct"
DEFAULT_DATASET = os.path.join(_SHARED_DIR, "goldilocks", "train.jsonl")
DEFAULT_CKPT_DIR = os.path.join(_SHARED_DIR, "phase2-checkpoints")
N_EVAL = 200


def load_eval_problems(dataset_path):
    """Load last N problems from dataset as held-out eval set."""
    with open(dataset_path) as f:
        lines = f.readlines()
    problems = []
    for line in lines[-N_EVAL:]:
        d = json.loads(line)
        problems.append((d["problem"], str(d["answer"])))
    return problems


def extract_boxed(text):
    results = []
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx == -1:
            break
        bs = idx + len("\\boxed{")
        depth = 1
        i = bs
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[bs:i - 1].strip())
        start = i
    return results[-1] if results else None


def check_answer(response, expected):
    """Check if model answer matches expected. Returns (correct, used_boxed)."""
    boxed = extract_boxed(response)
    used_boxed = boxed is not None

    # Method 1: boxed exact match
    if boxed is not None:
        if boxed.strip() == expected.strip():
            return True, True
        try:
            if abs(float(boxed) - float(expected)) < 0.01:
                return True, True
        except ValueError:
            pass

    # Method 2: last number in response
    nums = re.findall(r'-?\d+(?:\.\d+)?', response)
    if nums:
        try:
            if abs(float(nums[-1]) - float(expected)) < 0.01:
                return True, used_boxed
        except ValueError:
            pass

    # Method 3: substring
    if expected in response:
        return True, used_boxed

    return False, used_boxed


def evaluate(model, tokenizer, problems, label):
    """Run evaluation on a set of problems."""
    import torch

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    correct = 0
    boxed_count = 0
    per_problem = []
    t0 = time.time()

    for i, (problem, expected) in enumerate(problems):
        prompt = f"Solve this math problem step by step. Put your final answer in \\boxed{{}} tags.\n\n{problem}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        resp = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        is_correct, used_boxed = check_answer(resp, expected)

        if is_correct:
            correct += 1
        if used_boxed:
            boxed_count += 1

        per_problem.append({
            "problem": problem[:80],
            "expected": expected,
            "correct": is_correct,
            "boxed": used_boxed,
        })

        if i < 3 or is_correct:
            boxed_val = extract_boxed(resp) or "-"
            status = "OK" if is_correct else "FAIL"
            print(f"  [{i + 1:>2}] {status}  boxed={boxed_val:>8}  exp={expected:>8}")

    elapsed = time.time() - t0
    pct = 100 * correct / len(problems)

    print(f"\n  SCORE: {correct}/{len(problems)} ({pct:.0f}%)")
    print(f"  Used \\boxed{{}}: {boxed_count}/{len(problems)} ({100 * boxed_count // len(problems)}%)")
    print(f"  Time: {elapsed:.1f}s")

    return {
        "score": correct,
        "total": len(problems),
        "pct": pct,
        "boxed": boxed_count,
        "boxed_pct": 100 * boxed_count / len(problems),
        "per_problem": per_problem,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Nemotron-Mini-4B on Goldilocks math problems"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HuggingFace model name or local path")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Path to Goldilocks JSONL dataset")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CKPT_DIR,
                        help="Directory containing step_* LoRA checkpoints")
    parser.add_argument("--base-only", action="store_true",
                        help="Only evaluate the base model (skip checkpoint)")
    parser.add_argument("--output", default=None,
                        help="Path to write JSON results")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    problems = load_eval_problems(args.dataset)
    print("=" * 60)
    print(f"  Nemotron-Mini-4B Goldilocks Eval")
    print(f"  Model: {args.model}")
    print(f"  Eval: {len(problems)} held-out problems")
    print("=" * 60)

    # Base model eval
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    base_results = evaluate(model, tokenizer, problems, "BASE MODEL (before training)")

    if args.base_only:
        if args.output:
            with open(args.output, "w") as f:
                json.dump({"base": base_results}, f, indent=2)
        return

    # Find LoRA checkpoint
    del model
    torch.cuda.empty_cache()

    ckpt_path = Path(args.checkpoint_dir)
    steps = sorted(
        [d for d in ckpt_path.iterdir() if d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not steps:
        print(f"\nNo checkpoints in {args.checkpoint_dir}")
        return

    latest = steps[-1]
    lora_path = None
    for sub in [latest / "policy" / "weights" / "model", latest / "policy" / "weights", latest]:
        if (sub / "adapter_model.safetensors").exists():
            lora_path = str(sub)
            break

    if not lora_path:
        print(f"No adapter in {latest}")
        return

    print(f"\nLoading checkpoint: {lora_path}")

    # Fix adapter prefix
    cfg_file = Path(lora_path) / "adapter_config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            cfg = json.load(f)
        mods = cfg.get("target_modules", [])
        cleaned = [m.replace("_checkpoint_wrapped_module.", "") for m in mods]
        if cleaned != mods:
            cfg["target_modules"] = cleaned
            with open(cfg_file, "w") as f:
                json.dump(cfg, f, indent=2)
            print("  Fixed adapter prefix")

    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    model.eval()

    trained_results = evaluate(model, tokenizer, problems, f"TRAINED MODEL ({latest.name})")

    # Comparison
    print(f"\n{'=' * 60}")
    print(f"  BEFORE vs AFTER COMPARISON")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {base_results['score']}/{base_results['total']} ({base_results['pct']:.0f}%)  →  {trained_results['score']}/{trained_results['total']} ({trained_results['pct']:.0f}%)")
    print(f"  \\boxed{{}} : {base_results['boxed']}/{base_results['total']} ({base_results['boxed_pct']:.0f}%)  →  {trained_results['boxed']}/{trained_results['total']} ({trained_results['boxed_pct']:.0f}%)")

    acc_delta = trained_results["score"] - base_results["score"]
    box_delta = trained_results["boxed"] - base_results["boxed"]

    if acc_delta > 0:
        print(f"\n  Accuracy improved by {acc_delta} problems (+{acc_delta * 100 // base_results['total']}pp)")
    if box_delta > 0:
        print(f"  Format compliance improved by {box_delta} ({box_delta * 100 // base_results['total']}pp)")
    if acc_delta <= 0 and box_delta <= 0:
        print(f"\n  No improvement detected")
    print(f"{'=' * 60}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"base": base_results, "trained": trained_results}, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
