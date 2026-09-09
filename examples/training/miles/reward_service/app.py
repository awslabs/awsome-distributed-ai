# STATUS: Partly verified -- deployed on miles and exercised directly: /health, /ready, /metrics and
#   /score on both backends. math_verify returns 422 for an unusable gold label; a run of
#   service-side failures takes /ready to 503 and one success clears it, while bad data does not.
#   Not yet driven by miles's remote_rm as the reward path of a training run.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Remote Reward Service for miles GRPO training.

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
import multiprocessing
import os
import queue
import threading
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reward_service")

# Checked, because an unrecognised value silently left REWARD_BACKEND at "reward_model", which
# downloads a large classifier and scores against a reward nobody selected.
VALID_REWARD_BACKENDS = ("reward_model", "math_verify")
REWARD_BACKEND = os.environ.get("REWARD_BACKEND", "reward_model").strip()
REWARD_MODEL_NAME = os.environ.get(
    "REWARD_MODEL_NAME", "OpenAssistant/reward-model-deberta-v3-large-v2"
)
# Cap intra-op threads so a single replica does not monopolise the node; scale
# horizontally with replicas instead.
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "4"))
MAX_LENGTH = int(os.environ.get("REWARD_MAX_LENGTH", "2048"))


def _env_number(name: str, default: str, cast, minimum):
    # `or` rather than a get default: a manifest rendered with an unresolved variable delivers an
    # empty string, which a get default does not replace.
    raw = os.environ.get(name) or default
    try:
        value = cast(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}={raw!r} is not a number") from exc
    if not (value == value) or value in (float("inf"), float("-inf")):  # NaN / inf
        raise ValueError(f"{name}={raw!r} is not finite")
    if value < minimum:
        raise ValueError(f"{name}={raw!r} must be >= {minimum}")
    return value


# sympy on model-shaped text can run unbounded. The ceiling below is applied by the library to
# EACH of its three calls (gold parse, response parse, verify), while the parent waits the ceiling
# plus this grace for all three together. A verification that spends most of the ceiling in more
# than one call therefore hits the parent's clock first and is reported as a service timeout.
VERIFY_TIMEOUT_S = _env_number("REWARD_VERIFY_TIMEOUT_S", "10", float, 1)
VERIFY_PARENT_GRACE_S = _env_number("REWARD_VERIFY_PARENT_GRACE_S", "5", float, 1)
# One process each, so a wedged verification can be killed on its own.
VERIFY_SLOTS = int(_env_number("REWARD_VERIFY_SLOTS", "4", float, 1))
# Short on purpose: /score is a sync handler, so waiting here holds an anyio threadpool token and
# starves other /score calls. The probes are async and do not queue behind it.
VERIFY_SLOT_WAIT_S = _env_number("REWARD_VERIFY_SLOT_WAIT_S", "2", float, 0.1)
# Consecutive service-side failures before the pod reports itself not ready. Generous: a short
# burst must not cost a long run its reward service.
READY_MAX_CONSECUTIVE_FAILURES = _env_number("REWARD_READY_MAX_CONSECUTIVE_FAILURES", "20", int, 1)


class _Outcomes:
    """Scoring outcome counts, plus the consecutive-failure run that drives readiness.

    Status codes alone leave a persistently broken scorer behind a healthy-looking Deployment,
    because a retrying client turns the failure into latency rather than a bad pod. The counts make
    the state readable, and a run of consecutive service-side failures takes the pod out of the
    Service so the remaining replicas carry the load.

    A GoldLabelError does not count. It is one bad sample, not a sick service, and letting bad
    data evict pods would turn a dataset problem into an outage.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._counts = {}
        self._consecutive_failures = 0

    def ok(self):
        with self._lock:
            self._counts["scored_ok"] = self._counts.get("scored_ok", 0) + 1
            self._consecutive_failures = 0

    def service_failure(self, kind: str):
        with self._lock:
            self._counts[kind] = self._counts.get(kind, 0) + 1
            self._consecutive_failures += 1

    def bad_data(self, kind: str):
        with self._lock:
            self._counts[kind] = self._counts.get(kind, 0) + 1

    def consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive_failures

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._counts, consecutive_failures=self._consecutive_failures)


_outcomes = _Outcomes()


class GoldLabelError(Exception):
    """The label cannot be used as a gold answer. Bad data for one sample, not a sick service."""


class VerifierTimeoutError(Exception):
    """A verification exceeded its wall-clock ceiling."""


class VerifierBusyError(Exception):
    """No verifier slot was free: capacity, not breakage.

    It still counts toward the consecutive-failure run that drives readiness, so sustained load
    above VERIFY_SLOTS can take a healthy pod out of the Service.
    """


class ScoreRequest(BaseModel):
    prompt: str | list = ""
    response: str = ""
    # Labels are numeric in the JSONL. pydantic 2 rejects an int for a str field with 422, and
    # batched_async_rm gathers without return_exceptions, so one such label failed the whole
    # rollout batch. Normalised here rather than at each use site.
    label: str | int | float | None = None

    @field_validator("label", mode="after")
    @classmethod
    def _label_to_text(cls, v):
        # str() before any truthiness test: 0 is a legitimate gold answer.
        return None if v is None else str(v)


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


# Tags, not exception types: they cross the process boundary without needing picklable exception
# classes, and only the worker can tell sympy's bad-data ValueError from a real fault.
_OK = "ok"
_GOLD_ERROR = "gold_error"
_TIMEOUT = "timeout"
_FAILED = "failed"


def _math_verify_in_worker(label: str, response: str, timeout_s: float):
    """Run one verification, in a worker process.

    math_verify is imported here because each spawned worker starts with a fresh interpreter, and
    its own timeouts are passed through: signal.alarm() fires on a process's main thread, which is
    where this runs. Returns (tag, payload) rather than raising, so the outcome crosses the process
    boundary without depending on an exception class being picklable.
    """
    from math_verify import parse, verify
    from math_verify.errors import TimeoutException

    # int: the library times out with signal.alarm(). A float raises inside it, and
    # raise_on_error=False turns that into an empty parse, scoring every answer 0.0.
    lib_timeout = max(1, int(timeout_s))

    # raise_on_error=True separates the two: an empty parse is bad data, an exception is a fault.
    try:
        gold = parse(
            label if "\\boxed" in label else f"\\boxed{{{label}}}",
            parsing_timeout=lib_timeout,
            raise_on_error=True,
        )
    except TimeoutException as e:
        return (_TIMEOUT, f"gold parse: {e}")
    except Exception as e:  # noqa: BLE001 - the verifier itself failed
        return (_FAILED, f"gold parse: {type(e).__name__}: {e}")
    if not gold:
        return (_GOLD_ERROR, f"gold label did not parse: {label!r}")

    # raise_on_error=True here as well, so a parser fault is not silently scored 0.0. The cost is
    # that malformed model output reaches the caller as a service failure rather than as a wrong
    # answer.
    try:
        pred = parse(response, parsing_timeout=lib_timeout, raise_on_error=True)
    except TimeoutException as e:
        return (_TIMEOUT, f"response parse: {e}")
    except Exception as e:  # noqa: BLE001 - the verifier itself failed
        return (_FAILED, f"response parse: {type(e).__name__}: {e}")

    try:
        ok = verify(gold, pred, timeout_seconds=lib_timeout, raise_on_error=True)
    except TimeoutException as e:
        return (_TIMEOUT, f"verify: {e}")
    except Exception as e:  # noqa: BLE001
        return (_FAILED, f"{type(e).__name__}: {e}")
    return (_OK, 1.0 if ok else 0.0)


def _warmup_in_worker():
    """Pay the sympy import in a fresh worker before it is handed a timed request."""
    import math_verify  # noqa: F401

    return True


class _VerifierSlot:
    """One worker process, replaceable on its own.

    A pool-wide terminate would kill the healthy verifications beside the pathological one, so each
    slot owns a single-process pool. Pool.terminate() actually ends the child, where
    future.cancel() cannot stop a started task and shutdown() waits for it.
    """

    def __init__(self, ctx):
        self._ctx = ctx
        self._pool = None
        self.healthy = False
        self._spawn()

    def _spawn(self):
        self._pool = self._ctx.Pool(processes=1)
        # With spawn the child re-imports sympy, which on a cold filesystem can exceed the ceiling by
        # itself, so warmup gets its own budget rather than the request's. At startup nothing is
        # waiting on it; reached through recycle() after a timeout, the request pays it before its
        # 503, which is the one path where a caller waits longer than the ceiling suggests.
        self._pool.apply_async(_warmup_in_worker).get(timeout=max(60.0, VERIFY_TIMEOUT_S * 6))
        self.healthy = True

    def run(self, args, timeout_s: float):
        try:
            return self._pool.apply_async(_math_verify_in_worker, args).get(timeout=timeout_s)
        except multiprocessing.TimeoutError as exc:
            # Still inside sympy: kill it, or every later request on this slot inherits it.
            self.recycle()
            raise VerifierTimeoutError(f"verification exceeded {timeout_s}s") from exc
        except (EOFError, OSError, BrokenPipeError):
            # The child died or the pipe broke: the pool really is unusable.
            self.recycle()
            raise
        # Ordinary faults arrive as tags, so nothing here should cost a process: recycling on every
        # fault would let one malformed shard churn every slot.

    def recycle(self):
        self.healthy = False
        try:
            self._pool.terminate()
            self._pool.join()
        except Exception:  # noqa: BLE001 - already tearing down
            pass
        try:
            self._spawn()
        except Exception as e:  # noqa: BLE001 - stays unhealthy; the caller counts it out
            logger.error("could not respawn a verifier worker: %s", e)


class MathVerifyScorer(Scorer):
    """Rule-based math verification (LaTeX \\boxed{} + sympy) against label.

    Runs in a worker process under a wall-clock ceiling. In-process it cannot have one:
    math_verify's timeouts use signal.alarm(), which only fires on a process's main thread, and
    FastAPI dispatches sync handlers onto a threadpool.
    """

    def __init__(self):
        # Fail startup if the dependency is missing, rather than at the first request.
        import math_verify  # noqa: F401

        # spawn, not fork: the parent may already hold torch's threads and locks.
        self._ctx = multiprocessing.get_context("spawn")
        self._slots = queue.Queue()
        self._lock = threading.Lock()
        self._live = 0
        self._refill()

    def capacity(self) -> int:
        with self._lock:
            return self._live

    def _refill(self):
        """Create workers up to VERIFY_SLOTS. Called at startup and whenever one has been lost."""
        while self.capacity() < VERIFY_SLOTS:
            try:
                slot = _VerifierSlot(self._ctx)
            except Exception as e:  # noqa: BLE001 - capacity stays low; /ready reports it
                logger.error("could not create a verifier worker: %s", e)
                return
            with self._lock:
                self._live += 1
            self._slots.put(slot)

    def score(self, prompt, response: str, label: Optional[str]) -> float:
        # `label is None` rather than falsiness: "0" is a legitimate gold answer.
        if label is None or label == "":
            raise GoldLabelError("no label to score against")
        if self._slots.empty():
            self._refill()
        try:
            slot = self._slots.get(timeout=VERIFY_SLOT_WAIT_S)
        except queue.Empty as exc:
            raise VerifierBusyError(
                f"no free verifier slot within {VERIFY_SLOT_WAIT_S}s "
                f"({VERIFY_SLOTS} configured, {self.capacity()} live)"
            ) from exc
        try:
            tag, payload = slot.run(
                (label, response, VERIFY_TIMEOUT_S), VERIFY_TIMEOUT_S + VERIFY_PARENT_GRACE_S
            )
        finally:
            # A slot whose replacement could not be created must not go back on the queue.
            if slot.healthy:
                self._slots.put(slot)
            else:
                with self._lock:
                    self._live -= 1
                logger.error("verifier worker lost; %d of %d live", self.capacity(), VERIFY_SLOTS)
        if tag == _OK:
            return payload
        if tag == _GOLD_ERROR:
            raise GoldLabelError(payload)
        if tag == _TIMEOUT:
            raise VerifierTimeoutError(payload)
        raise RuntimeError(payload)


def _build_scorer() -> Scorer:
    if REWARD_BACKEND not in VALID_REWARD_BACKENDS:
        # Fail startup rather than default: no reward this service returns would be the one the
        # run asked for.
        raise ValueError(
            f"REWARD_BACKEND={REWARD_BACKEND!r} is not one of {VALID_REWARD_BACKENDS}."
        )
    if REWARD_BACKEND == "math_verify":
        logger.info("Using math_verify backend.")
        return MathVerifyScorer()
    logger.info("Using reward_model backend: %s", REWARD_MODEL_NAME)
    return RewardModelScorer(REWARD_MODEL_NAME)


app = FastAPI(title="miles Remote Reward Service")
_scorer: Optional[Scorer] = None


@app.on_event("startup")
def _startup():
    global _scorer
    _scorer = _build_scorer()


# async so they never wait for the threadpool token /score holds.
@app.get("/health")
async def health():
    """Liveness: the process is up and the scorer was built."""
    return {"status": "ok", "backend": REWARD_BACKEND}


@app.get("/ready")
async def ready(response: Response):
    """Readiness: a verifier is available and scoring is not failing in a run.

    Readiness rather than liveness, deliberately. Wiring failures into the liveness probe would
    crash-loop the pod on one deterministic bad input mid-rollout; taking it out of the Service
    leaves the remaining replicas serving. Note that a pod removed from the Service stops receiving
    the successful score that would reset the run, so past the threshold it needs a restart or a
    direct request.
    """
    capacity = _scorer.capacity() if hasattr(_scorer, "capacity") else None
    if _scorer is None or capacity == 0:
        response.status_code = 503
        return {"status": "no verifier capacity", "backend": REWARD_BACKEND}
    failures = _outcomes.consecutive_failures()
    if failures >= READY_MAX_CONSECUTIVE_FAILURES:
        response.status_code = 503
        return {"status": "scoring is failing", "backend": REWARD_BACKEND,
                "consecutive_failures": failures}
    return {"status": "ok", "backend": REWARD_BACKEND, "verifier_slots": capacity,
            "consecutive_failures": failures}


@app.get("/metrics")
async def metrics():
    """Scoring outcomes, so a run producing no rewards is visible while it is still running."""
    return {"backend": REWARD_BACKEND, **_outcomes.snapshot()}


@app.post("/score")
def score(req: ScoreRequest):
    """Return the reward, or an HTTP error when there is no reward to return.

    remote_rm assigns the returned JSON value directly to sample.reward, and 0.0 is a legitimate
    reward, so a failure answered with 0.0 is a failure the run trains on. An error status says
    instead that this sample was not scored. A client that retries turns a transient fault into
    latency rather than lost correctness; one that does not will fail the batch, which is still
    better than training on a reward nobody computed. Which of the two miles is is unconfirmed.

    422 for a label that parses to nothing, because retrying will not change it; a parser fault
    on the label is a service failure and answers 503 like the rest;
    503 for a timeout, a busy verifier or an internal fault, because retrying might.
    """
    try:
        value = _scorer.score(req.prompt, req.response, req.label)
    except GoldLabelError as e:
        _outcomes.bad_data("gold_label_error")
        logger.warning("gold label unusable: %s", e)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except VerifierTimeoutError as e:
        _outcomes.service_failure("verify_timeout")
        logger.warning("verification timed out: %s", e)
        raise HTTPException(status_code=503, detail=str(e)) from e
    except VerifierBusyError as e:
        _outcomes.service_failure("slot_exhausted")
        logger.warning("no verifier capacity: %s", e)
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001 - reported as a service fault, not as a reward
        _outcomes.service_failure("internal_error")
        logger.exception("scoring failed")
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}") from e
    _outcomes.ok()
    return value
