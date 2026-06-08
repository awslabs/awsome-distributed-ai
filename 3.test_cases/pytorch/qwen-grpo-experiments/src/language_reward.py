# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Language compliance reward function for veRL GRPO training.

Scores model completions based on language compliance:
  - Answer language match:    +5.0 / -5.0
  - Reasoning language match: +1.5 / -1.5
  - Answer brevity (<=2 sent): +0.5 / -1.0

Total range: [-7.5, +7.0]

veRL interface: compute_score(data_source, solution_str, ground_truth, extra_info)
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    logger.warning("langdetect not installed — reward function will return 0.0")


def extract_reasoning(response: str) -> str:
    """Extract reasoning section from response."""
    for pattern in [
        r'<think>(.*?)</think>',
        r'<thinking>(.*?)</thinking>',
        r'<analysis>(.*?)</analysis>',
    ]:
        m = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    # Fallback: first 70% of response as reasoning
    cutoff = int(len(response) * 0.7)
    return response[:cutoff] if response else ""


def extract_final_answer(response: str) -> str:
    """Extract final answer section from response."""
    for pattern in [
        r'</think>\s*(.*?)$',
        r'</thinking>\s*(.*?)$',
        r'<answer>(.*?)</answer>',
    ]:
        m = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    # Fallback: last 30% of response
    cutoff = int(len(response) * 0.7)
    return response[cutoff:] if response else ""


def detect_language(text: str) -> str:
    """Detect language, returns ISO 639-1 code."""
    if not HAS_LANGDETECT:
        return "unknown"
    try:
        clean = re.sub(r'[0-9\+\-\*\/\=\%\$]', '', text)
        clean = re.sub(r'[^\w\s]', ' ', clean)
        if len(clean.strip()) < 20:
            return "too_short"
        return detect(clean)
    except Exception:
        return "error"


def count_sentences(text: str) -> int:
    """Count sentences."""
    sentences = re.split(r'[.!?]+', text.strip())
    return len([s for s in sentences if s.strip()])


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """
    veRL-compatible reward function for language compliance.

    Args:
        data_source: Dataset identifier (unused).
        solution_str: Model's generated completion.
        ground_truth: JSON with {"expected_lang": "es", "language_name": "Spanish"}.
        extra_info: Optional metadata (unused).

    Returns:
        Float reward in [-7.5, +7.0].
    """
    if not HAS_LANGDETECT:
        return 0.0

    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        expected_code = gt.get("expected_lang", "en")
    except (json.JSONDecodeError, TypeError, AttributeError):
        return 0.0

    if not solution_str:
        return -7.5

    reasoning = extract_reasoning(solution_str)
    final_answer = extract_final_answer(solution_str)
    reasoning_lang = detect_language(reasoning)
    answer_lang = detect_language(final_answer)

    reward = 0.0

    # 1. Answer language (70% weight)
    reward += 5.0 if answer_lang == expected_code else -5.0

    # 2. Reasoning language (20% weight)
    reward += 1.5 if reasoning_lang == expected_code else -1.5

    # 3. Brevity (10% weight)
    reward += 0.5 if count_sentences(final_answer) <= 2 else -1.0

    return reward
