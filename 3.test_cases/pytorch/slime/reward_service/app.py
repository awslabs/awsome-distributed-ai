# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Remote Reward Service for SLIME GRPO training.

This is a CPU-hosted HTTP reward server that implements the contract expected by
SLIME's ``remote_rm`` hook (slime/rollout/rm_hub/__init__.py):

    POST {RM_URL}
        body: {"prompt": str, "response": str, "label": str | null}
        returns: a bare JSON number (the scalar reward), which SLIME assigns
                 directly to ``sample.reward``.

Why run this off the GPU nodes?
    The GPU rollout engines (SGLang) and trainers (Megatron) are the expensive,
    scarce resource. Scoring should not steal their CPU. A reward *model* (a
    small sequence classifier) or any heavy verifier (code-exec, RAG, unit
    tests) is CPU/IO-bound and latency-tolerant, so it belongs on a cheap,
    independently-scalable CPU instance group in the same AZ. The reward RPC is
    low-bandwidth HTTP and does NOT use EFA/RDMA.

Backends (select with REWARD_BACKEND):
    - "reward_model" (default): a HuggingFace AutoModelForSequenceClassification
      that scores (prompt, response) pairs on CPU. This is the case where the
      CPU offload is a genuine throughput win.
    - "math_verify": rule-based LaTeX/sympy verification against ``label``;
      useful as a zero-dependency fallback / for math datasets.
"""

import logging
import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reward_service")

REWARD_BACKEND = os.environ.get("REWARD_BACKEND", "reward_model").strip()
REWARD_MODEL_NAME = os.environ.get(
    "REWARD_MODEL_NAME", "OpenAssistant/reward-model-deberta-v3-large-v2"
)
# Cap intra-op threads so a single replica does not monopolise the node; scale
# horizontally with replicas instead.
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "4"))
MAX_LENGTH = int(os.environ.get("REWARD_MAX_LENGTH", "2048"))


class ScoreRequest(BaseModel):
    prompt: str | list = ""
    response: str = ""
    label: Optional[str] = None


# --------------------------------------------------------------------------- #
# Pluggable scorer backends
# --------------------------------------------------------------------------- #
class Scorer:
    """Backend interface: score a single (prompt, response, label) -> float."""

    def score(self, prompt: str, response: str, label: Optional[str]) -> float:
        raise NotImplementedError


class RewardModelScorer(Scorer):
    """HuggingFace sequence-classifier reward model running on CPU."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(TORCH_NUM_THREADS)
        self.torch = torch
        logger.info("Loading reward model %s on CPU ...", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
        self.model.eval()
        logger.info("Reward model loaded.")

    def _to_text(self, prompt) -> str:
        if isinstance(prompt, list):
            # chat-format prompt: concatenate message contents
            return "\n".join(
                m.get("content", "") for m in prompt if isinstance(m, dict)
            )
        return prompt or ""

    def score(self, prompt, response: str, label: Optional[str]) -> float:
        prompt_text = self._to_text(prompt)
        with self.torch.no_grad():
            inputs = self.tokenizer(
                prompt_text,
                response,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
            )
            logits = self.model(**inputs).logits
            # Single-logit reward models output a scalar score; multi-class
            # models -> take the positive/last class logit.
            score = logits.squeeze(-1) if logits.shape[-1] == 1 else logits[..., -1]
            return float(score.reshape(-1)[0].item())


class MathVerifyScorer(Scorer):
    """Rule-based math verification (LaTeX \\boxed{} + sympy) against label."""

    def __init__(self):
        from math_verify import parse, verify

        self._parse = parse
        self._verify = verify

    def score(self, prompt, response: str, label: Optional[str]) -> float:
        if not label:
            return 0.0
        try:
            # parsing_timeout=None: math_verify's default timeout uses
            # signal.alarm(), which only works on the main thread. FastAPI runs
            # sync handlers in a threadpool, so disable the signal-based timeout.
            gold = self._parse(
                label if "\\boxed" in str(label) else f"\\boxed{{{label}}}",
                parsing_timeout=None,
            )
            pred = self._parse(response, parsing_timeout=None)
            return 1.0 if self._verify(gold, pred, timeout_seconds=None) else 0.0
        except Exception as e:  # noqa: BLE001 - verifier must never crash the service
            logger.warning("math_verify scoring error: %s", e)
            return 0.0


def _build_scorer() -> Scorer:
    if REWARD_BACKEND == "math_verify":
        logger.info("Using math_verify backend.")
        return MathVerifyScorer()
    logger.info("Using reward_model backend: %s", REWARD_MODEL_NAME)
    return RewardModelScorer(REWARD_MODEL_NAME)


app = FastAPI(title="SLIME Remote Reward Service")
_scorer: Optional[Scorer] = None


@app.on_event("startup")
def _startup():
    global _scorer
    _scorer = _build_scorer()


@app.get("/health")
def health():
    return {"status": "ok", "backend": REWARD_BACKEND}


@app.post("/score")
def score(req: ScoreRequest):
    # SLIME's remote_rm assigns the returned JSON value directly to
    # sample.reward, so we return a bare float.
    return _scorer.score(req.prompt, req.response, req.label)
