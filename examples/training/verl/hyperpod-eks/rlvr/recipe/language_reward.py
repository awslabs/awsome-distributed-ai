# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Custom reward function for veRL GRPO training.

Scores model completions on language compliance for the Multilingual-Thinking
task.  The model is instructed (via the system prompt) to reason AND answer in
a specific language.  This reward function checks:

  1. Answer language match   (+5.0 / -5.0)  — most important
  2. Reasoning language match (+1.5 / -1.5)
  3. Final answer brevity    (+0.5 / -1.0)  — ≤2 sentences

Max score: +7.0   Min score: -7.5

Function signature follows the veRL custom_reward_function API:
    compute_score(data_source, solution_str, ground_truth, extra_info) -> dict

The ``ground_truth`` field is expected to contain the target language code
(e.g. ``"en"``, ``"fr"``).  See ``load_data_gptoss.sh`` for how the parquet
data is prepared.
"""

import re

try:
    from langdetect import detect, DetectorFactory

    DetectorFactory.seed = 0  # deterministic
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LANG_CODE_MAP = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
}


def _extract_reasoning(response: str) -> str:
    """Return the text between the analysis and final-answer markers."""
    m = re.search(
        r"assistantanalysis\s*(.*?)\s*assistantfinal",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return response[:500] if len(response) > 500 else response


def _extract_final_answer(response: str) -> str:
    """Return everything after the final-answer marker."""
    m = re.search(
        r"assistantfinal\s*(.*)$",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return response[-200:] if len(response) > 200 else response


def _detect_language(text: str) -> str:
    if not HAS_LANGDETECT:
        return "unknown"
    try:
        clean = re.sub(r"[0-9+\-*/=%$€£]", "", text)
        clean = re.sub(r"[^\w\s]", " ", clean)
        if len(clean.strip()) < 20:
            return "too_short"
        return detect(clean)
    except Exception:
        return "error"


def _count_sentences(text: str) -> int:
    parts = re.split(r"[.!?]+", text.strip())
    return len([s for s in parts if s.strip()])


# ---------------------------------------------------------------------------
# veRL entry-point
# ---------------------------------------------------------------------------


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Score a single completion for language compliance.

    Parameters
    ----------
    data_source : str
        Dataset identifier (unused here — kept for API compat).
    solution_str : str
        The model's decoded completion text.
    ground_truth : str
        The expected language **code** (e.g. ``"en"``, ``"fr"``).
        Stored in the parquet ``reward_model.ground_truth`` column.
    extra_info : dict, optional
        Not used by this reward function.

    Returns
    -------
    dict
        ``{"score": float, "answer_lang": str, "reasoning_lang": str,
          "answer_correct": bool, "reasoning_correct": bool}``
    """
    expected_code = ground_truth if ground_truth else "en"

    reasoning = _extract_reasoning(solution_str)
    final_answer = _extract_final_answer(solution_str)
    reasoning_lang = _detect_language(reasoning)
    answer_lang = _detect_language(final_answer)

    reward = 0.0

    # 1. Answer language — 70 % of signal
    answer_correct = answer_lang == expected_code
    reward += 5.0 if answer_correct else -5.0

    # 2. Reasoning language — 20 % of signal
    reasoning_correct = reasoning_lang == expected_code
    reward += 1.5 if reasoning_correct else -1.5

    # 3. Brevity — 10 % of signal
    n_sentences = _count_sentences(final_answer)
    reward += 0.5 if n_sentences <= 2 else -1.0

    return {
        "score": reward,
        "answer_lang": answer_lang,
        "reasoning_lang": reasoning_lang,
        "answer_correct": answer_correct,
        "reasoning_correct": reasoning_correct,
    }
