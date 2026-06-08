# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Simplified binary language reward for veRL GRPO training.

Simpler alternative to language_reward.py:
  - Correct answer language: +1.0
  - Wrong answer language:   -1.0

This reduces reward hacking risk and provides cleaner learning signal.

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


def detect_language(text: str) -> str:
    """Detect language of text."""
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


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """
    Binary language compliance reward.

    Returns +1.0 if the response is in the expected language, -1.0 otherwise.
    """
    if not HAS_LANGDETECT:
        return 0.0

    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        expected_code = gt.get("expected_lang", "en")
    except (json.JSONDecodeError, TypeError, AttributeError):
        return 0.0

    if not solution_str or len(solution_str.strip()) < 10:
        return -1.0

    detected = detect_language(solution_str)
    return 1.0 if detected == expected_code else -1.0
