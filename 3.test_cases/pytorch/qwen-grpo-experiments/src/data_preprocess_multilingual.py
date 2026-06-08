# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Data preprocessing for Phase 1: Multilingual Language Compliance.

Converts HuggingFaceH4/Multilingual-Thinking into veRL-compatible parquet:
  - Detects question language
  - Applies chat template with language instructions
  - Stores ground truth for reward function

Output: train.parquet, val.parquet with columns:
  - prompt: Chat-templated prompt string
  - reward_model/ground_truth: JSON {"expected_lang": "es", "language_name": "Spanish"}

Usage:
    python data_preprocess_multilingual.py \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --output_dir /fsx/qwen-grpo/data/multilingual \
        --eval_size 50
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    logger.warning("langdetect not installed — defaulting to English")

LANG_CODE_MAP = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
}
CODE_TO_NAME = {v: k for k, v in LANG_CODE_MAP.items()}
SUPPORTED_CODES = set(LANG_CODE_MAP.values())


def detect_question_language(text: str) -> str:
    """Detect language of question text, return ISO 639-1 code."""
    if not HAS_LANGDETECT:
        return "en"
    try:
        import re
        clean = re.sub(r'[0-9\+\-\*\/\=\%\$\€\£]', '', text)
        clean = re.sub(r'[^\w\s]', ' ', clean)
        if len(clean.strip()) < 10:
            return "en"
        code = detect(clean)
        return code if code in SUPPORTED_CODES else "en"
    except Exception:
        return "en"


def format_prompt_with_template(tokenizer, question: str, language_name: str) -> str:
    """Format prompt using model's chat template."""
    system_msg = (
        f"You are a helpful assistant. "
        f"Reason in {language_name}. Answer in {language_name}. "
        f"Keep your final answer concise (1-2 sentences)."
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
        return format_prompt_manual(question, language_name)


def format_prompt_manual(question: str, language_name: str) -> str:
    """Fallback prompt formatting without tokenizer."""
    return (
        f"<|im_start|>system\n"
        f"You are a helpful assistant. "
        f"Reason in {language_name}. Answer in {language_name}. "
        f"Keep your final answer concise (1-2 sentences).<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def process_example(question: str, tokenizer=None) -> dict:
    """Process a single example into veRL format."""
    lang_code = detect_question_language(question)
    language_name = CODE_TO_NAME.get(lang_code, "English")

    if tokenizer is not None:
        prompt = format_prompt_with_template(tokenizer, question, language_name)
    else:
        prompt = format_prompt_manual(question, language_name)

    ground_truth = json.dumps({
        "expected_lang": lang_code,
        "language_name": language_name,
    })

    return {
        "prompt": prompt,
        "reward_model/ground_truth": ground_truth,
    }


def preprocess_dataset(dataset, tokenizer=None, eval_size: int = 50, max_samples: int = None):
    """
    Preprocess HuggingFace dataset into veRL format.

    Returns (train_df, val_df) DataFrames.
    """
    train_split = dataset["train"] if "train" in dataset else dataset

    records = []
    for i, example in enumerate(train_split):
        if max_samples and i >= max_samples:
            break
        messages = example.get("messages", [])
        question = ""
        for msg in messages:
            if msg["role"] == "user":
                question = msg["content"]
                break
        if not question:
            continue
        records.append(process_example(question, tokenizer))

    df = pd.DataFrame(records)

    # Split: stratified by language if possible
    actual_eval = min(eval_size, len(df) // 5)
    val_df = df.iloc[:actual_eval].reset_index(drop=True)
    train_df = df.iloc[actual_eval:].reset_index(drop=True)

    return train_df, val_df


def validate_parquet(df: pd.DataFrame, name: str) -> bool:
    """Validate DataFrame meets veRL requirements."""
    required_cols = {"prompt", "reward_model/ground_truth"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error(f"[{name}] Missing columns: {missing}")
        return False
    if len(df) == 0:
        logger.error(f"[{name}] DataFrame is empty")
        return False
    null_counts = df[list(required_cols)].isnull().sum()
    if null_counts.any():
        logger.error(f"[{name}] Null values: {null_counts.to_dict()}")
        return False
    empty_prompts = (df["prompt"].astype(str).str.strip() == "").sum()
    if empty_prompts > 0:
        logger.error(f"[{name}] {empty_prompts} empty prompts")
        return False
    return True


def log_language_distribution(df: pd.DataFrame, name: str):
    """Log language distribution."""
    lang_counts = Counter()
    for gt_str in df["reward_model/ground_truth"]:
        try:
            gt = json.loads(gt_str)
            lang_counts[gt.get("expected_lang", "unknown")] += 1
        except (json.JSONDecodeError, TypeError):
            lang_counts["parse_error"] += 1
    print(f"[{name}] Language distribution: {dict(lang_counts)}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess multilingual data for veRL GRPO")
    parser.add_argument("--dataset_name", type=str,
                        default="HuggingFaceH4/Multilingual-Thinking")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str,
                        default="/fsx/qwen-grpo/data/multilingual")
    parser.add_argument("--eval_size", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap total samples (for debugging)")
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

    # Load dataset
    from datasets import load_dataset
    print(f"Loading dataset: {args.dataset_name}")
    try:
        dataset = load_dataset(args.dataset_name)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # Preprocess
    print(f"Preprocessing (eval_size={args.eval_size}, max_samples={args.max_samples})...")
    train_df, val_df = preprocess_dataset(dataset, tokenizer, args.eval_size, args.max_samples)

    # Validate
    if not validate_parquet(train_df, "train") or not validate_parquet(val_df, "val"):
        print("ERROR: Validation failed")
        sys.exit(1)

    log_language_distribution(train_df, "train")
    log_language_distribution(val_df, "val")
    print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")

    # SFT format (messages column for veRL SFT trainer)
    print("Creating SFT-format parquet...")
    sft_records = []
    train_split = dataset["train"] if "train" in dataset else dataset
    for i, example in enumerate(train_split):
        if args.max_samples and i >= args.max_samples:
            break
        messages = example.get("messages", [])
        if not messages:
            continue
        sft_records.append({"messages": json.dumps(messages)})

    sft_df = pd.DataFrame(sft_records)
    actual_eval = min(args.eval_size, len(sft_df) // 5)
    sft_train = sft_df.iloc[actual_eval:].reset_index(drop=True)
    sft_val = sft_df.iloc[:actual_eval].reset_index(drop=True)

    sft_train_path = os.path.join(args.output_dir, "sft_train.parquet")
    sft_val_path = os.path.join(args.output_dir, "sft_val.parquet")
    sft_train.to_parquet(sft_train_path, index=False)
    sft_val.to_parquet(sft_val_path, index=False)
    print(f"Saved SFT: {sft_train_path} ({len(sft_train)} rows)")
    print(f"Saved SFT: {sft_val_path} ({len(sft_val)} rows)")

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
