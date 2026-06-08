# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Math reasoning reward function for veRL GRPO training.

Scores model completions based on:
  1. Correct final answer (exact match): +1.0 / -1.0
  2. Format compliance (has \\boxed{}):  +0.2 / -0.2

Total range: [-1.2, +1.2]

Uses math_verify library when available for robust answer comparison.

veRL interface: compute_score(data_source, solution_str, ground_truth, extra_info)
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

try:
    from math_verify import verify, parse
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False
    logger.warning("math_verify not installed — using exact string match")


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{...} in response."""
    # Handle nested braces
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if matches:
        return matches[-1].strip()  # Last boxed answer

    # Fallback: #### format (GSM8K style)
    match = re.search(r'####\s*([\-\d,\.]+)', text)
    if match:
        return match.group(1).replace(",", "").strip()

    # Fallback: last number in response
    numbers = re.findall(r'[\-]?\d[\d,]*\.?\d*', text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.strip()
    # Remove trailing zeros after decimal
    try:
        num = float(answer.replace(",", ""))
        if num == int(num):
            return str(int(num))
        return str(num)
    except (ValueError, OverflowError):
        return answer.lower().strip()


def answers_match(predicted: str, expected: str) -> bool:
    """Check if predicted answer matches expected answer."""
    if HAS_MATH_VERIFY:
        try:
            return verify(parse(predicted), parse(expected))
        except Exception:
            pass  # Fall through to string comparison

    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)

    if pred_norm == exp_norm:
        return True

    # Numeric comparison with tolerance
    try:
        pred_num = float(pred_norm)
        exp_num = float(exp_norm)
        return abs(pred_num - exp_num) < 1e-6
    except (ValueError, OverflowError):
        return False


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """
    Math reasoning reward function.

    Args:
        data_source: Dataset identifier (unused).
        solution_str: Model's generated completion.
        ground_truth: JSON with {"answer": "42", "dataset": "gsm8k"}.
        extra_info: Optional metadata (unused).

    Returns:
        Float reward in [-1.2, +1.2].
    """
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        expected_answer = gt.get("answer", "")
    except (json.JSONDecodeError, TypeError, AttributeError):
        return 0.0

    if not solution_str or not expected_answer:
        return -1.2

    # Extract predicted answer
    predicted = extract_boxed_answer(solution_str)

    reward = 0.0

    # 1. Correctness (+/- 1.0)
    if predicted and answers_match(predicted, expected_answer):
        reward += 1.0
    else:
        reward -= 1.0

    # 2. Format compliance: has \boxed{} (+/- 0.2)
    if re.search(r'\\boxed\{', solution_str):
        reward += 0.2
    else:
        reward -= 0.2

    return reward
