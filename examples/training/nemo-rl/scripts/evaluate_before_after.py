#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Before/after evaluation for GRPO workshop demo.

Uses the SAME prompt format as training (CoT with \\boxed{}) so the
evaluation measures what GRPO actually teaches the model.

Usage:
  python3 evaluate_before_after.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --checkpoint-dir /shared/nvrx-demo/checkpoints
"""

import argparse
import re
import sys
import time
import json
from pathlib import Path


# ── Prompt template (same as NeMo RL training) ──
COT_TEMPLATE = (
    "Think step-by-step to solve the following problem. "
    "Output your answer inside of \\boxed{{}} tags.:\n{}\n\n"
    "Let's think step-by-step"
)

# ── Evaluation problems with known answers ──
EVAL_PROMPTS = [
    ("Arithmetic", "What is 247 + 389?", "636"),
    ("Word problem", "A store sells apples for $2 each and oranges for $3 each. If Sarah buys 5 apples and 3 oranges, how much does she spend in total?", "19"),
    ("Fractions", "What is 3/4 + 1/6? Simplify your answer.", "11/12"),
    ("Algebra", "Solve for x: 2x + 7 = 15", "4"),
    ("Percentage", "What is 15% of 240?", "36"),
    ("Equation", "If 3x - 5 = 16, what is x?", "7"),
    ("Geometry", "What is the area of a triangle with base 12 cm and height 8 cm?", "48"),
    ("Multi-step", "A train travels at 60 km/h for 2.5 hours, then at 80 km/h for 1.5 hours. What is the total distance traveled in km?", "270"),
    ("Division", "If you divide 156 by 12, what do you get?", "13"),
    ("Roots", "What is the square root of 144?", "12"),
]


def extract_boxed(text):
    """Extract answer from \\boxed{...} in model response."""
    # Find all \boxed{...} patterns, take the last one (final answer)
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()
    return None


def check_answer(response, expected):
    """Check if the model's answer matches expected, trying multiple methods."""
    resp_lower = response.lower().strip()
    exp_lower = expected.lower().strip()

    # Method 1: Extract from \boxed{} (what GRPO trains the model to do)
    boxed = extract_boxed(response)
    if boxed is not None:
        # Normalize: strip $, spaces, trailing .0
        boxed_clean = boxed.replace("$", "").replace(",", "").strip()
        exp_clean = exp_lower.replace("$", "").replace(",", "").strip()
        if boxed_clean == exp_clean:
            return True, "boxed"
        # Try numeric comparison
        try:
            if abs(float(boxed_clean) - float(exp_clean)) < 0.01:
                return True, "boxed-numeric"
        except ValueError:
            pass

    # Method 2: Direct substring (fallback for models that don't use \boxed{})
    if exp_lower in resp_lower:
        return True, "substring"

    # Method 3: Numeric anywhere in response
    try:
        exp_num = float(exp_lower.replace("$", "").replace(",", ""))
        # Look for the number in the response
        for token in resp_lower.replace(",", "").split():
            token = token.strip(".$()[]")
            try:
                if abs(float(token) - exp_num) < 0.01:
                    return True, "numeric"
            except ValueError:
                continue
    except ValueError:
        pass

    return False, "none"


def load_model_and_tokenizer(model_name, lora_path=None):
    """Load model with optional LoRA weights."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if lora_path:
        print(f"Loading LoRA weights from: {lora_path}")
        try:
            from peft import PeftModel
        except ImportError:
            print("peft not installed — installing now...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "peft", "-q"])
            from peft import PeftModel
        try:
            # Fix activation checkpoint prefix
            adapter_config_path = Path(lora_path) / "adapter_config.json"
            if adapter_config_path.exists():
                with open(adapter_config_path) as f:
                    adapter_cfg = json.load(f)
                target_modules = adapter_cfg.get("target_modules", [])
                cleaned = [m.replace("_checkpoint_wrapped_module.", "") for m in target_modules]
                if cleaned != target_modules:
                    print(f"  Fixing adapter config prefix ({len(target_modules)} modules)")
                    adapter_cfg["target_modules"] = cleaned
                    with open(adapter_config_path, "w") as f:
                        json.dump(adapter_cfg, f, indent=2)
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
            print("LoRA weights merged successfully")
        except Exception as e:
            print(f"WARNING: Could not load LoRA weights: {e}")
            print("Falling back to base model")

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, problem, max_new_tokens=512):
    """Generate a response using the CoT prompt template."""
    import torch

    # Use the same prompt format as GRPO training
    prompt = COT_TEMPLATE.format(problem)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        start = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        elapsed = time.time() - start

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip(), elapsed


def run_evaluation(model, tokenizer, label):
    """Run all evaluation problems and return results."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")

    results = []
    for i, (category, problem, expected) in enumerate(EVAL_PROMPTS):
        print(f"[{i+1}/{len(EVAL_PROMPTS)}] {category}")

        response, elapsed = generate_response(model, tokenizer, problem)

        correct, method = check_answer(response, expected)
        boxed = extract_boxed(response)

        # Show key info
        boxed_str = f"\\boxed{{{boxed}}}" if boxed else "(no \\boxed)"
        print(f"  Answer: {boxed_str}  Expected: {expected}  {'CORRECT' if correct else 'WRONG'}  ({elapsed:.1f}s)")
        print(f"  Response: {response[:200]}{'...' if len(response) > 200 else ''}")
        print()

        results.append({
            "category": category,
            "problem": problem,
            "response": response,
            "boxed_answer": boxed,
            "expected": expected,
            "correct": correct,
            "method": method,
            "time_s": round(elapsed, 2),
        })

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"Score: {correct}/{total} ({100*correct/total:.0f}%)")

    # Show boxed format adoption
    boxed_count = sum(1 for r in results if r["boxed_answer"] is not None)
    print(f"Used \\boxed{{}}: {boxed_count}/{total}")

    return results


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest step_N checkpoint directory."""
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        return None

    step_dirs = sorted(
        [d for d in ckpt_path.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
        reverse=True,
    )

    if not step_dirs:
        return None

    latest = step_dirs[0]
    model_subdir = latest / "policy" / "weights" / "model"
    if (model_subdir / "adapter_model.safetensors").exists():
        return str(model_subdir)
    weights_dir = latest / "policy" / "weights"
    if (weights_dir / "adapter_model.safetensors").exists():
        return str(weights_dir)
    if (latest / "adapter_model.safetensors").exists():
        return str(latest)
    if weights_dir.exists():
        return str(weights_dir)
    return str(latest)


def print_comparison(base_results, trained_results):
    """Print side-by-side comparison."""
    print(f"\n{'='*60}")
    print(f"  BEFORE vs AFTER Training Comparison")
    print(f"{'='*60}\n")

    print(f"{'Category':<20} {'Before':>10} {'After':>10} {'Change':>10}")
    print("-" * 55)

    base_correct = 0
    trained_correct = 0
    base_boxed = 0
    trained_boxed = 0

    for b, t in zip(base_results, trained_results):
        b_status = "CORRECT" if b["correct"] else "WRONG"
        t_status = "CORRECT" if t["correct"] else "WRONG"
        change = ""
        if not b["correct"] and t["correct"]:
            change = "IMPROVED"
        elif b["correct"] and not t["correct"]:
            change = "REGRESSED"
        else:
            change = ""

        base_correct += int(b["correct"])
        trained_correct += int(t["correct"])
        base_boxed += int(b["boxed_answer"] is not None)
        trained_boxed += int(t["boxed_answer"] is not None)

        print(f"{b['category']:<20} {b_status:>10} {t_status:>10} {change:>10}")

    print("-" * 55)
    print(f"{'ACCURACY':<20} {base_correct:>7}/{len(base_results)} {trained_correct:>7}/{len(trained_results)}")
    print(f"{'USED \\boxed{}':<20} {base_boxed:>7}/{len(base_results)} {trained_boxed:>7}/{len(trained_results)}")

    print(f"\n  Math accuracy: {base_correct}/{len(base_results)} -> {trained_correct}/{len(trained_results)}")
    print(f"  Format (\\boxed{{}}): {base_boxed}/{len(base_results)} -> {trained_boxed}/{len(trained_results)}")

    improvement = trained_correct - base_correct
    if improvement > 0:
        print(f"\n  Training improved {improvement} answer(s)!")
    elif improvement == 0:
        if trained_boxed > base_boxed:
            print(f"\n  Same accuracy but model learned \\boxed{{}} format ({base_boxed} -> {trained_boxed})")
        else:
            print(f"\n  Same score — try more training steps")
    else:
        print(f"\n  Regression on {-improvement} answer(s) — may need different hyperparameters")


def main():
    parser = argparse.ArgumentParser(description="Before/after GRPO evaluation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base model name")
    parser.add_argument("--checkpoint-dir", default="/shared/nvrx-demo/checkpoints",
                        help="Checkpoint directory on FSx")
    parser.add_argument("--base-only", action="store_true",
                        help="Only evaluate base model (before training)")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    print("=" * 60)
    print("  GRPO Training Evaluation — Before & After")
    print(f"  Model: {args.model}")
    print(f"  Checkpoints: {args.checkpoint_dir}")
    print(f"  Prompt: CoT with \\boxed{{}} (same as training)")
    print("=" * 60)

    # ── Evaluate base model ──
    model, tokenizer = load_model_and_tokenizer(args.model)
    base_results = run_evaluation(model, tokenizer, "BEFORE TRAINING (Base Model)")

    if args.base_only:
        print("\n(--base-only specified, skipping trained model evaluation)")
        return

    # ── Find and load trained checkpoint ──
    lora_path = find_latest_checkpoint(args.checkpoint_dir)
    if not lora_path:
        print(f"\nNo checkpoints found in {args.checkpoint_dir}")
        print("Run training first, then re-run this evaluation.")
        return

    print(f"\nFound checkpoint: {lora_path}")

    # Reload with LoRA
    del model
    import torch
    torch.cuda.empty_cache()

    model, tokenizer = load_model_and_tokenizer(args.model, lora_path=lora_path)
    trained_results = run_evaluation(model, tokenizer, "AFTER TRAINING (GRPO + LoRA)")

    # ── Comparison ──
    print_comparison(base_results, trained_results)

    # ── Save results ──
    if args.output:
        output = {
            "model": args.model,
            "checkpoint": lora_path,
            "base_results": base_results,
            "trained_results": trained_results,
            "base_score": sum(1 for r in base_results if r["correct"]),
            "trained_score": sum(1 for r in trained_results if r["correct"]),
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
