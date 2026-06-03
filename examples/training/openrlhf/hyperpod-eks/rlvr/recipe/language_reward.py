# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Custom reward function for OpenRLHF GRPO training.

Scores model completions on language compliance for the Multilingual-Thinking
task.  The model is instructed (via the system prompt) to reason AND answer in
a specific language.  This reward function checks:

  1. Answer language match   — most important signal
  2. Reasoning language match
  3. Final answer brevity    — ≤2 sentences

OpenRLHF API contract
---------------------
Function MUST be named ``reward_func`` and return a dict with::

    {"rewards": Tensor, "scores": Tensor, "extra_logs": dict}

``queries``  — decoded prompt+response strings (special tokens included)
``prompts``  — original prompt strings (after chat template)
``labels``   — ground truth from ``--label_key`` in dataset

OpenRLHF calls this function with **batch_size=1** per vLLM response.

The ``labels`` value is the expected language **code** (e.g. ``"en"``,
``"fr"``), stored in the dataset under the ``label`` key.
"""

import re

import torch

try:
    from langdetect import detect, DetectorFactory

    DetectorFactory.seed = 0  # deterministic
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False


# ---------------------------------------------------------------------------
# Helpers (identical logic to the veRL reward function)
# ---------------------------------------------------------------------------

LANG_CODE_MAP = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
}


def _extract_response(query: str, prompt: str) -> str:
    """Extract the model's response by removing the prompt prefix."""
    # OpenRLHF passes queries = prompt + response (with special tokens).
    # The simplest extraction is to strip the prompt prefix.
    if query.startswith(prompt):
        return query[len(prompt):]
    # Fallback: look for assistant turn markers
    # Try common patterns
    for marker in ["<|assistant|>", "<|im_start|>assistant", "assistant\n"]:
        idx = query.rfind(marker)
        if idx >= 0:
            return query[idx + len(marker):]
    # Last resort: return last 60% of text
    return query[len(query) // 3:]


def _extract_reasoning(response: str) -> str:
    """Return the text between the analysis and final-answer markers."""
    m = re.search(
        r"assistantanalysis\s*(.*?)\s*assistantfinal",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    # Fallback patterns
    for pattern in [
        r"<analysis>(.*?)</analysis>",
        r"<thinking>(.*?)</thinking>",
    ]:
        m = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return response[:500] if len(response) > 500 else response


def _extract_final_answer(response: str) -> str:
    """Return everything after the final-answer marker."""
    m = re.search(
        r"assistantfinal\s*(.*?)$",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    for pattern in [r"<final>(.*?)</final>"]:
        m = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
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


def _score_single(response: str, expected_code: str) -> dict:
    """Score a single completion for language compliance.

    Returns dict with score and diagnostic fields.
    Max score: +7.0   Min score: -7.5
    """
    reasoning = _extract_reasoning(response)
    final_answer = _extract_final_answer(response)
    reasoning_lang = _detect_language(reasoning)
    answer_lang = _detect_language(final_answer)

    reward = 0.0

    # 1. Answer language — 70% of signal
    answer_correct = answer_lang == expected_code
    reward += 5.0 if answer_correct else -5.0

    # 2. Reasoning language — 20% of signal
    reasoning_correct = reasoning_lang == expected_code
    reward += 1.5 if reasoning_correct else -1.5

    # 3. Brevity — 10% of signal
    n_sentences = _count_sentences(final_answer)
    reward += 0.5 if n_sentences <= 2 else -1.0

    return {
        "score": reward,
        "answer_lang": answer_lang,
        "reasoning_lang": reasoning_lang,
        "answer_correct": answer_correct,
        "reasoning_correct": reasoning_correct,
    }


# ---------------------------------------------------------------------------
# OpenRLHF entry-point — MUST be named reward_func
# ---------------------------------------------------------------------------


def reward_func(queries, prompts, labels, **kwargs):
    """Score completions for language compliance.

    Parameters
    ----------
    queries : list[str]
        Decoded full sequences (prompt + response, special tokens included).
    prompts : list[str]
        Original prompt strings (after chat template application).
    labels : list[str]
        Ground truth language codes (e.g. "en", "fr") from ``--label_key``.

    Returns
    -------
    dict
        {"rewards": Tensor, "scores": Tensor, "extra_logs": dict}
    """
    rewards = []
    answer_correct_count = 0
    reasoning_correct_count = 0

    for query, prompt, label in zip(queries, prompts, labels):
        response = _extract_response(query, prompt)
        expected_code = label if label else "en"
        result = _score_single(response, expected_code)
        rewards.append(result["score"])
        if result["answer_correct"]:
            answer_correct_count += 1
        if result["reasoning_correct"]:
            reasoning_correct_count += 1

    n = len(queries)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float)

    # Normalize scores to [0, 1] for dynamic filtering compatibility
    # score range is [-7.5, 7.0] → map to [0, 1]
    scores_tensor = (rewards_tensor + 7.5) / 14.5

    return {
        "rewards": rewards_tensor,
        "scores": scores_tensor,
        "extra_logs": {
            "answer_accuracy": torch.tensor([answer_correct_count / max(n, 1)]),
            "reasoning_accuracy": torch.tensor([reasoning_correct_count / max(n, 1)]),
            "mean_reward": rewards_tensor.mean().unsqueeze(0),
        },
    }
