# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Data preprocessing for Phase 2: Math Reasoning (GSM8K + MATH).

Converts GSM8K and optionally MATH dataset into veRL-compatible parquet:
  - Extracts numeric answers for reward verification
  - Applies Qwen chat template with CoT instructions
  - Stores ground truth for math_reward.py

Output: train.parquet, val.parquet with columns:
  - prompt: Chat-templated prompt string
  - reward_model/ground_truth: JSON {"answer": "42", "dataset": "gsm8k"}

Usage:
    python data_preprocess_math.py \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --output_dir /fsx/qwen-grpo/data/math \
        --datasets gsm8k,math
"""

import argparse
import json
import logging
import os
import re
import sys

import pandas as pd

logger = logging.getLogger(__name__)


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract numeric answer from GSM8K format: '#### <number>'."""
    match = re.search(r'####\s*([\-\d,\.]+)', answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    # Fallback: last number in text
    numbers = re.findall(r'[\-]?\d[\d,]*\.?\d*', answer_text)
    return numbers[-1].replace(",", "") if numbers else ""


def extract_math_answer(solution: str) -> str:
    """Extract answer from MATH format: \\boxed{answer}."""
    match = re.search(r'\\boxed\{([^}]+)\}', solution)
    if match:
        return match.group(1).strip()
    return ""


def format_math_prompt(tokenizer, question: str) -> str:
    """Format math prompt with CoT instruction."""
    system_msg = (
        "You are a helpful math assistant. "
        "Think step by step to solve the problem. "
        "Put your final numeric answer inside \\boxed{}."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


def format_sft_messages(question: str, solution: str) -> list:
    """Format as messages for SFT training."""
    system_msg = (
        "You are a helpful math assistant. "
        "Think step by step to solve the problem. "
        "Put your final numeric answer inside \\boxed{}."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question},
        {"role": "assistant", "content": solution},
    ]


def process_gsm8k(tokenizer, max_samples: int = None):
    """Process GSM8K dataset."""
    from datasets import load_dataset

    print("Loading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main")

    grpo_records = []
    sft_records = []

    for i, example in enumerate(ds["train"]):
        if max_samples and i >= max_samples:
            break

        question = example["question"]
        answer_text = example["answer"]
        numeric_answer = extract_gsm8k_answer(answer_text)

        if not numeric_answer:
            continue

        prompt = format_math_prompt(tokenizer, question)
        ground_truth = json.dumps({
            "answer": numeric_answer,
            "dataset": "gsm8k",
        })

        grpo_records.append({
            "prompt": prompt,
            "reward_model/ground_truth": ground_truth,
        })

        # SFT format: include the full solution
        sft_messages = format_sft_messages(question, answer_text)
        sft_records.append({"messages": json.dumps(sft_messages)})

    # Test split
    grpo_test = []
    sft_test = []
    for example in ds["test"]:
        question = example["question"]
        answer_text = example["answer"]
        numeric_answer = extract_gsm8k_answer(answer_text)
        if not numeric_answer:
            continue

        prompt = format_math_prompt(tokenizer, question)
        ground_truth = json.dumps({"answer": numeric_answer, "dataset": "gsm8k"})
        grpo_test.append({"prompt": prompt, "reward_model/ground_truth": ground_truth})
        sft_messages = format_sft_messages(question, answer_text)
        sft_test.append({"messages": json.dumps(sft_messages)})

    return grpo_records, sft_records, grpo_test, sft_test


def process_math_dataset(tokenizer, max_samples: int = None):
    """Process MATH dataset (hendrycks/competition_math)."""
    from datasets import load_dataset

    print("Loading MATH dataset...")
    try:
        ds = load_dataset("hendrycks/competition_math")
    except Exception as e:
        logger.warning(f"Could not load MATH dataset: {e}")
        return [], [], [], []

    grpo_records = []
    sft_records = []

    split = ds["train"] if "train" in ds else list(ds.values())[0]
    for i, example in enumerate(split):
        if max_samples and i >= max_samples:
            break

        question = example.get("problem", "")
        solution = example.get("solution", "")
        answer = extract_math_answer(solution)

        if not answer or not question:
            continue

        prompt = format_math_prompt(tokenizer, question)
        ground_truth = json.dumps({"answer": answer, "dataset": "math"})

        grpo_records.append({
            "prompt": prompt,
            "reward_model/ground_truth": ground_truth,
        })
        sft_messages = format_sft_messages(question, solution)
        sft_records.append({"messages": json.dumps(sft_messages)})

    # Test split
    grpo_test = []
    sft_test = []
    test_split = ds.get("test", ds.get("validation", []))
    for example in test_split:
        question = example.get("problem", "")
        solution = example.get("solution", "")
        answer = extract_math_answer(solution)
        if not answer or not question:
            continue
        prompt = format_math_prompt(tokenizer, question)
        ground_truth = json.dumps({"answer": answer, "dataset": "math"})
        grpo_test.append({"prompt": prompt, "reward_model/ground_truth": ground_truth})
        sft_messages = format_sft_messages(question, solution)
        sft_test.append({"messages": json.dumps(sft_messages)})

    return grpo_records, sft_records, grpo_test, sft_test


def main():
    parser = argparse.ArgumentParser(description="Preprocess math data for veRL GRPO")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str,
                        default="/fsx/qwen-grpo/data/math")
    parser.add_argument("--datasets", type=str, default="gsm8k",
                        help="Comma-separated: gsm8k,math")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--eval_size", type=int, default=200,
                        help="Samples from train to use as additional validation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load tokenizer
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.warning(f"Could not load tokenizer ({e}), using manual formatting")

    datasets_to_process = [d.strip() for d in args.datasets.split(",")]

    all_grpo_train = []
    all_sft_train = []
    all_grpo_test = []
    all_sft_test = []

    if "gsm8k" in datasets_to_process:
        grpo_r, sft_r, grpo_t, sft_t = process_gsm8k(tokenizer, args.max_samples)
        all_grpo_train.extend(grpo_r)
        all_sft_train.extend(sft_r)
        all_grpo_test.extend(grpo_t)
        all_sft_test.extend(sft_t)
        print(f"GSM8K: {len(grpo_r)} train, {len(grpo_t)} test")

    if "math" in datasets_to_process:
        grpo_r, sft_r, grpo_t, sft_t = process_math_dataset(tokenizer, args.max_samples)
        all_grpo_train.extend(grpo_r)
        all_sft_train.extend(sft_r)
        all_grpo_test.extend(grpo_t)
        all_sft_test.extend(sft_t)
        print(f"MATH: {len(grpo_r)} train, {len(grpo_t)} test")

    print(f"\nTotal: {len(all_grpo_train)} train, {len(all_grpo_test)} test")

    # Create DataFrames
    train_df = pd.DataFrame(all_grpo_train)
    test_df = pd.DataFrame(all_grpo_test)
    sft_train_df = pd.DataFrame(all_sft_train)
    sft_test_df = pd.DataFrame(all_sft_test)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    train_df.to_parquet(os.path.join(args.output_dir, "train.parquet"), index=False)
    test_df.to_parquet(os.path.join(args.output_dir, "test.parquet"), index=False)
    sft_train_df.to_parquet(os.path.join(args.output_dir, "sft_train.parquet"), index=False)
    sft_test_df.to_parquet(os.path.join(args.output_dir, "sft_val.parquet"), index=False)

    print(f"\nSaved to {args.output_dir}:")
    print(f"  train.parquet: {len(train_df)} rows (GRPO)")
    print(f"  test.parquet: {len(test_df)} rows (GRPO eval)")
    print(f"  sft_train.parquet: {len(sft_train_df)} rows (SFT)")
    print(f"  sft_val.parquet: {len(sft_test_df)} rows (SFT eval)")


if __name__ == "__main__":
    main()
