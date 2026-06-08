# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Math evaluation script for Phase 2.

Evaluates model accuracy on GSM8K test set and optionally MATH test set.
Uses vLLM for fast batch inference.

Usage:
    python evaluate_math.py \
        --model_path Qwen/Qwen2.5-7B-Instruct \
        --adapter_path /fsx/qwen-grpo/checkpoints/math/sft \
        --output_file /fsx/qwen-grpo/results/math_eval.json \
        --dataset gsm8k
"""

import argparse
import json
import os
import re
import sys
import time

import torch


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{...}."""
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if matches:
        return matches[-1].strip()
    # GSM8K format
    match = re.search(r'####\s*([\-\d,\.]+)', text)
    if match:
        return match.group(1).replace(",", "").strip()
    # Last number
    numbers = re.findall(r'[\-]?\d[\d,]*\.?\d*', text)
    return numbers[-1].replace(",", "") if numbers else ""


def normalize_answer(answer: str) -> str:
    """Normalize for comparison."""
    answer = answer.strip()
    try:
        num = float(answer.replace(",", ""))
        if num == int(num):
            return str(int(num))
        return str(num)
    except (ValueError, OverflowError):
        return answer.lower().strip()


def answers_match(predicted: str, expected: str) -> bool:
    """Check answer match."""
    try:
        from math_verify import verify, parse
        try:
            return verify(parse(predicted), parse(expected))
        except Exception:
            pass
    except ImportError:
        pass

    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)
    if pred_norm == exp_norm:
        return True
    try:
        return abs(float(pred_norm) - float(exp_norm)) < 1e-6
    except (ValueError, OverflowError):
        return False


def load_test_data(dataset: str, max_samples: int = None):
    """Load test data."""
    from datasets import load_dataset

    examples = []

    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for i, ex in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            match = re.search(r'####\s*([\-\d,\.]+)', ex["answer"])
            if match:
                examples.append({
                    "question": ex["question"],
                    "answer": match.group(1).replace(",", "").strip(),
                    "dataset": "gsm8k",
                })
    elif dataset == "math":
        ds = load_dataset("hendrycks/competition_math", split="test")
        for i, ex in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            match = re.search(r'\\boxed\{([^}]+)\}', ex.get("solution", ""))
            if match:
                examples.append({
                    "question": ex["problem"],
                    "answer": match.group(1).strip(),
                    "dataset": "math",
                })

    return examples


def evaluate_with_vllm(model_path: str, adapter_path: str, examples: list, model_name: str):
    """Evaluate using vLLM for fast inference."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM not available, falling back to HF generate")
        return evaluate_with_hf(model_path, adapter_path, examples, model_name)

    print(f"Loading vLLM: {model_path}")
    llm_kwargs = {
        "model": model_path,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "max_model_len": 4096,
    }
    if adapter_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 32

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(temperature=0.0, max_tokens=2048, top_p=1.0)

    # Build prompts
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompts = []
    for ex in examples:
        system_msg = (
            "You are a helpful math assistant. "
            "Think step by step to solve the problem. "
            "Put your final numeric answer inside \\boxed{}."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": ex["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    print(f"Generating {len(prompts)} responses...")
    if adapter_path:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("adapter", 1, adapter_path)
        outputs = llm.generate(prompts, sampling, lora_request=lora_req)
    else:
        outputs = llm.generate(prompts, sampling)

    # Score
    correct = 0
    results_detail = []
    for i, (output, ex) in enumerate(zip(outputs, examples)):
        response = output.outputs[0].text
        predicted = extract_boxed_answer(response)
        is_correct = answers_match(predicted, ex["answer"])
        if is_correct:
            correct += 1
        results_detail.append({
            "question": ex["question"][:100],
            "expected": ex["answer"],
            "predicted": predicted,
            "correct": is_correct,
            "response_preview": response[:300],
        })

    accuracy = correct / len(examples) * 100
    return {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": examples[0]["dataset"] if examples else "unknown",
        "correct": correct,
        "total": len(examples),
        "accuracy": accuracy,
        "details": results_detail,
    }


def evaluate_with_hf(model_path: str, adapter_path: str, examples: list, model_name: str):
    """Fallback evaluation using HuggingFace generate."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    if adapter_path and os.path.exists(adapter_path):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()

    correct = 0
    results_detail = []

    for i, ex in enumerate(examples):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(examples)}")

        system_msg = (
            "You are a helpful math assistant. "
            "Think step by step. Put your final answer inside \\boxed{}."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": ex["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.0, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        predicted = extract_boxed_answer(response)
        is_correct = answers_match(predicted, ex["answer"])
        if is_correct:
            correct += 1
        results_detail.append({
            "question": ex["question"][:100],
            "expected": ex["answer"],
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / len(examples) * 100
    return {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": examples[0]["dataset"] if examples else "unknown",
        "correct": correct,
        "total": len(examples),
        "accuracy": accuracy,
        "details": results_detail,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate math reasoning accuracy")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--output_file", type=str,
                        default="/fsx/qwen-grpo/results/math_eval.json")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "math"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--use_vllm", action="store_true", default=True)
    args = parser.parse_args()

    model_name = args.model_name or args.model_path
    if args.adapter_path:
        model_name += f" + {os.path.basename(args.adapter_path)}"

    print(f"=== Math Evaluation: {model_name} ===")
    print(f"Dataset: {args.dataset}")

    examples = load_test_data(args.dataset, args.max_samples)
    print(f"Test examples: {len(examples)}")

    if args.use_vllm:
        results = evaluate_with_vllm(args.model_path, args.adapter_path, examples, model_name)
    else:
        results = evaluate_with_hf(args.model_path, args.adapter_path, examples, model_name)

    print(f"\n=== {args.dataset.upper()} Accuracy: {results['accuracy']:.1f}% "
          f"({results['correct']}/{results['total']}) ===")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {args.output_file}")


if __name__ == "__main__":
    main()
