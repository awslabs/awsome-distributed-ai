# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Batch evaluation runner for all experiment arms.

Runs evaluation on all checkpoints and produces a summary comparison table.

Usage:
    python batch_eval.py --phase multilingual --results_dir /fsx/qwen-grpo/results
    python batch_eval.py --phase math --results_dir /fsx/qwen-grpo/results
    python batch_eval.py --phase all --results_dir /fsx/qwen-grpo/results
"""

import argparse
import json
import os
import subprocess
import sys
import time


# Arm definitions for each phase
PHASE1_ARMS = {
    "A_base": {
        "description": "Qwen2.5-7B-Instruct (no fine-tuning)",
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "adapter_path": None,
    },
    "B_sft": {
        "description": "SFT only (multilingual)",
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "adapter_path": "/fsx/qwen-grpo/checkpoints/multilingual/sft",
    },
    "C_grpo": {
        "description": "GRPO only (from base)",
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "adapter_path": "/fsx/qwen-grpo/checkpoints/multilingual/grpo-from-base",
    },
    "D_sft_grpo": {
        "description": "SFT + GRPO (GRPO from SFT checkpoint)",
        "model_path": "/fsx/qwen-grpo/models/qwen-sft-multilingual-merged",
        "adapter_path": "/fsx/qwen-grpo/checkpoints/multilingual/grpo-from-sft",
    },
}

PHASE2_ARMS = {
    "A_base": {
        "description": "Qwen2.5-7B-Instruct (no fine-tuning)",
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "adapter_path": None,
    },
    "B_sft": {
        "description": "SFT only (math)",
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "adapter_path": "/fsx/qwen-grpo/checkpoints/math/sft",
    },
    "C_grpo": {
        "description": "GRPO only (from base)",
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "adapter_path": "/fsx/qwen-grpo/checkpoints/math/grpo-from-base",
    },
    "D_sft_grpo": {
        "description": "SFT + GRPO (GRPO from SFT checkpoint)",
        "model_path": "/fsx/qwen-grpo/models/qwen-sft-math-merged",
        "adapter_path": "/fsx/qwen-grpo/checkpoints/math/grpo-from-sft",
    },
}


def run_eval(arm_name: str, arm_config: dict, phase: str, results_dir: str, dataset: str = None):
    """Run evaluation for a single arm."""
    output_file = os.path.join(results_dir, f"{phase}_{arm_name}.json")

    if os.path.exists(output_file):
        print(f"  [SKIP] {arm_name}: results already exist at {output_file}")
        with open(output_file) as f:
            return json.load(f)

    # Check if adapter exists (skip if not trained yet)
    if arm_config["adapter_path"] and not os.path.exists(arm_config["adapter_path"]):
        print(f"  [SKIP] {arm_name}: adapter not found at {arm_config['adapter_path']}")
        return None

    if phase == "multilingual":
        cmd = [
            sys.executable, "-m", "evaluate_multilingual",
            "--model_path", arm_config["model_path"],
            "--output_file", output_file,
            "--model_name", f"{arm_name}: {arm_config['description']}",
        ]
        if arm_config["adapter_path"]:
            cmd.extend(["--adapter_path", arm_config["adapter_path"]])
    else:
        cmd = [
            sys.executable, "-m", "evaluate_math",
            "--model_path", arm_config["model_path"],
            "--output_file", output_file,
            "--model_name", f"{arm_name}: {arm_config['description']}",
            "--dataset", dataset or "gsm8k",
        ]
        if arm_config["adapter_path"]:
            cmd.extend(["--adapter_path", arm_config["adapter_path"]])

    print(f"  [RUN] {arm_name}: {arm_config['description']}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            print(f"  [ERROR] {arm_name}: {result.stderr[:500]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {arm_name}")
        return None

    if os.path.exists(output_file):
        with open(output_file) as f:
            return json.load(f)
    return None


def print_summary_table(all_results: dict, phase: str):
    """Print formatted comparison table."""
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY: Phase {phase.upper()}")
    print(f"{'='*70}")
    print(f"{'Arm':<12} {'Description':<35} {'Accuracy':>10}")
    print(f"{'-'*12} {'-'*35} {'-'*10}")

    for arm_name, result in sorted(all_results.items()):
        if result is None:
            accuracy = "N/A"
        else:
            accuracy = f"{result.get('overall', result).get('accuracy', 0):.1f}%"
        desc = PHASE1_ARMS.get(arm_name, PHASE2_ARMS.get(arm_name, {})).get("description", "")
        print(f"{arm_name:<12} {desc:<35} {accuracy:>10}")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for all arms")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["multilingual", "math", "all"])
    parser.add_argument("--results_dir", type=str,
                        default="/fsx/qwen-grpo/results")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        help="Math dataset: gsm8k or math")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.phase in ("multilingual", "all"):
        print("\n=== Phase 1: Multilingual Evaluation ===")
        results = {}
        for arm_name, arm_config in PHASE1_ARMS.items():
            results[arm_name] = run_eval(arm_name, arm_config, "multilingual", args.results_dir)
        print_summary_table(results, "multilingual")

        # Save summary
        summary_path = os.path.join(args.results_dir, "phase1_summary.json")
        with open(summary_path, "w") as f:
            json.dump({k: v for k, v in results.items() if v}, f, indent=2)

    if args.phase in ("math", "all"):
        print("\n=== Phase 2: Math Evaluation ===")
        results = {}
        for arm_name, arm_config in PHASE2_ARMS.items():
            results[arm_name] = run_eval(
                arm_name, arm_config, "math", args.results_dir, args.dataset
            )
        print_summary_table(results, "math")

        summary_path = os.path.join(args.results_dir, "phase2_summary.json")
        with open(summary_path, "w") as f:
            json.dump({k: v for k, v in results.items() if v}, f, indent=2)


if __name__ == "__main__":
    main()
