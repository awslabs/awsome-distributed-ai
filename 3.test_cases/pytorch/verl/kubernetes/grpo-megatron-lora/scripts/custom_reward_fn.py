# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# Custom Reward Function — SandboxFusion URL Injection + Eurus Fallback
#
# Two responsibilities:
#
# 1. Ensures sandbox_fusion_url and memory_limit_mb are reliably passed to
#    default_compute_score() for code tasks (apps, taco, codecontests), avoiding
#    fallback to prime_code which imports pyext (crashes on Python 3.12).
#
# 2. Handles data_source values not recognized by verl's default_compute_score.
#    The Eurus-2-RL-Data dataset (PRIME-RL) has per-sample data_source values
#    like "numina_olympiads", "taco", etc. that verl natively supports.
#    However, if data was prepared with an older script that set data_source
#    to a blanket "eurus" or "PRIME-RL/Eurus-2-RL-Data", verl raises
#    NotImplementedError.  The fallback below detects math vs code from the
#    ground_truth format and routes accordingly.
#
# Compatible with verl main (0.8.0.dev, commit b7249af) and v0.7.0.
# The import path verl.utils.reward_score.default_compute_score is unchanged.
# When using use_reward_loop=False (our configuration), verl calls
# custom_reward_function directly — the Reward Loop's reward_router_address
# kwarg is absorbed by **kwargs and safely ignored.
#
# Loaded via the custom_reward_function config:
#
#   custom_reward_function.path=/workspace/custom_reward_fn.py
#   custom_reward_function.name=compute_score
#   custom_reward_function.reward_kwargs.sandbox_fusion_url=http://...
#   custom_reward_function.reward_kwargs.memory_limit_mb=1024
#
# The function signature matches verl's compute_score interface exactly.
#
# Tracking: https://github.com/verl-project/verl/issues/4318 (Reward Loop RFC)
# =============================================================================

import json
import logging
import os
import threading
import time

from verl.utils.reward_score import default_compute_score, sandbox_fusion

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_REWARD_LOG_LEVEL", "INFO"))

# Counter for diagnostic logging — log first N samples per data_source to
# avoid flooding logs.  Set VERL_REWARD_LOG_SAMPLES=0 to disable.
_MAX_LOG_SAMPLES = int(os.getenv("VERL_REWARD_LOG_SAMPLES", "5"))
_log_counts: dict[str, int] = {}


# -----------------------------------------------------------------------------
# Log-flood guard for verl's sandbox_fusion scorer.
#
# verl/utils/reward_score/sandbox_fusion/utils.py logs the ENTIRE generated
# code and the ENTIRE test-case stdin at ERROR level for every failed sandbox
# call ("Case N: code: ...", "Case N: input: ...").  Competitive-programming
# test inputs can be 100k+ lines of numbers, so a burst of sandbox timeouts
# floods the Ray driver log (2026-07-12 incident: 99.9997% of a 198MB log
# tail was this spam, burying all training metrics).
#
# This module is imported inside every reward worker, so installing a
# truncating filter here tames that logger everywhere.  Error visibility is
# preserved — messages are just capped.  Tune via env:
#   VERL_SANDBOX_LOG_MAX_CHARS  max chars per log record (default 500, 0=off)
# -----------------------------------------------------------------------------
_SANDBOX_LOG_MAX_CHARS = int(os.getenv("VERL_SANDBOX_LOG_MAX_CHARS", "500"))


class _TruncatingFilter(logging.Filter):
    def __init__(self, max_chars: int):
        super().__init__()
        self.max_chars = max_chars

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            if len(msg) > self.max_chars:
                record.msg = msg[: self.max_chars] + f" ...[truncated {len(msg) - self.max_chars} chars]"
                record.args = ()
        except Exception:  # noqa: BLE001 — never let log filtering break scoring
            pass
        return True


if _SANDBOX_LOG_MAX_CHARS > 0:
    logging.getLogger("verl.utils.reward_score.sandbox_fusion.utils").addFilter(
        _TruncatingFilter(_SANDBOX_LOG_MAX_CHARS)
    )


# Data sources that verl's default_compute_score handles natively.
# If the sample's data_source is one of these, pass straight through.
_KNOWN_DATA_SOURCES = {
    # Math
    "openai/gsm8k",
    "lighteval/MATH",
    "DigitalLearningGmbH/MATH-lighteval",
    "HuggingFaceH4/MATH-500",
    "math_dapo",
    "math",
    "math_dapo_reasoning",
    "numina_aops_forum",
    "numina_synthetic_math",
    "numina_amc_aime",
    "numina_synthetic_amc",
    "numina_cn_k12",
    "numina_olympiads",
    # Code
    "codecontests",
    "apps",
    "codeforces",
    "taco",
    # Vision / QA
    "hiyouga/geometry3k",
    "searchR1_nq",
    "searchR1_triviaqa",
    "searchR1_popqa",
    "searchR1_hotpotqa",
    "searchR1_2wikimultihopqa",
    "searchR1_musique",
    "searchR1_bamboogle",
}

# Data sources whose scoring goes through Sandbox Fusion (i.e. subject to the
# sandbox-failure scoring bug described below).
# Must match the "# Code" block above.
_CODE_DATA_SOURCES = frozenset({"codecontests", "apps", "codeforces", "taco"})


# =============================================================================
# Held-out val rows: distinct METRIC key, identical SCORING path
# =============================================================================
# The MATH val aggregate has n=515, giving it a +-0.060 margin of error -- the
# same size as the effect being measured (+0.0661 at step 100). Enlarging it is
# the cheapest resolution win available, so a
# holdout split of never-trained rows is appended to the val parquet.
#
# Those rows need a data_source that verl reports SEPARATELY, because verl keys
# validation metrics as `val-core/<data_source>/acc/mean@1` and the legacy
# 515-row aggregate must stay computable for continuity with prior runs. Hence a
# suffix: `numina_cn_k12` (legacy, 216 rows) vs `numina_cn_k12_h2` (holdout).
#
# BUT a suffixed name is not in _KNOWN_DATA_SOURCES, so it would take the
# `fallback-math` path and be scored as `numina_olympiads`. That is PROBABLY the
# same prime_math scorer -- and "probably the same scorer" is how this project
# acquired eleven silent instrument bugs. A holdout row scored by even a
# slightly different code path than its legacy counterpart makes the two
# aggregates incomparable, silently, in the direction nobody would check.
#
# So the suffix is stripped for ROUTING only. `numina_cn_k12_h2` routes exactly
# as `numina_cn_k12` does -- native, same effective_source, same scorer -- while
# the original label still reaches verl's metric key and the [REWARD] logger.
# Stripping is deliberately conservative: it applies ONLY when the base name is
# itself a known source, so an unrelated source that happens to end in "_h2"
# cannot be mangled into something else.
_HOLDOUT_SUFFIX = "_h2"


def _routing_source(data_source):
    """Map a val-holdout data_source onto the base source it must be scored as.

    Returns data_source unchanged for every non-holdout label, so this is a no-op
    for all training data and all pre-existing val rows.
    """
    if (
        isinstance(data_source, str)
        and data_source.endswith(_HOLDOUT_SUFFIX)
        and data_source[: -len(_HOLDOUT_SUFFIX)] in _KNOWN_DATA_SOURCES
    ):
        return data_source[: -len(_HOLDOUT_SUFFIX)]
    return data_source


def _resolve_route(data_source, ground_truth):
    """Decide (effective_source, route) for a sample. Pure: no I/O, no verl import.

    Extracted from compute_score so it can be unit-tested offline -- the routing
    decision is the part that must be provably identical between a legacy row and
    its holdout twin, and that property is worth a test rather than a comment.
    """
    routing_source = _routing_source(data_source)

    if routing_source in _KNOWN_DATA_SOURCES or routing_source.startswith("aime"):
        # Fast path: natively supported
        return routing_source, "native"
    if _is_code_ground_truth(ground_truth):
        # Code task — route to sandbox execution (same as "apps"/"taco")
        return "apps", "fallback-code"
    # Math task — route to prime_math (same as "numina_olympiads")
    return "numina_olympiads", "fallback-math"


def _is_code_ground_truth(ground_truth):
    """Detect whether ground_truth contains code test cases (JSON with inputs/outputs).

    Code tasks in Eurus have ground_truth like:
        {"inputs": ["1 2\n", "3 4\n"], "outputs": ["3\n", "7\n"]}
    Math tasks have plain string answers like "42" or "\\frac{1}{2}".
    """
    if not isinstance(ground_truth, str):
        # Already a dict — check for test case structure
        if isinstance(ground_truth, dict):
            return "inputs" in ground_truth or "outputs" in ground_truth
        return False
    try:
        parsed = json.loads(ground_truth)
        if isinstance(parsed, dict) and ("inputs" in parsed or "outputs" in parsed):
            return True
    except (json.JSONDecodeError, TypeError):
        pass
    return False


# =============================================================================
# Sandbox Fusion concurrency cap
# =============================================================================
# verl's sandbox path accepts a `concurrent_semaphore` and, when given one, wraps
# every execute-API call in it (verl/utils/reward_score/sandbox_fusion/utils.py).
# It defaults to None == UNBOUNDED, and unbounded concurrency OOM-kills the
# sandbox pods.
#
# This semaphore bounds in-flight REQUESTS, not executions. One request fans out
# to ~39 sandbox executions, and nothing bounds those — so lowering this reduces
# the blast radius of a pod death but does NOT fix sandbox OOM. It is per-process
# and verl runs several RewardLoopWorkers, so total concurrency is workers x this
# value.
#
# Do NOT respond to sandbox OOM by raising pod memory limits or adding replicas:
# the pods share a node with ray-head, and evicting ray-head kills the training
# job outright. See docs/results.md ("Sandbox concurrency and OOM") for the
# measurements behind both conclusions.
_SANDBOX_MAX_CONCURRENT = int(os.environ.get("SANDBOX_MAX_CONCURRENT_PER_WORKER", "5"))
_sandbox_semaphore = (
    threading.Semaphore(_SANDBOX_MAX_CONCURRENT) if _SANDBOX_MAX_CONCURRENT > 0 else None
)


# =============================================================================
# Sandbox INFRASTRUCTURE failures were scored as WRONG ANSWERS
# =============================================================================
# verl's sandbox scorer DOES have an error channel. It returns a tuple:
#
#   verl/utils/reward_score/sandbox_fusion/__init__.py
#     def compute_score(url, semaphore, memory_limit_mb, completion,
#                       test_cases, continuous=False, timeout=10)
#         -> (float score, list[dict] metadata)
#
# and each metadata entry carries a "status" that distinguishes an unreachable
# sandbox from a genuinely wrong answer:
#
#   INFRASTRUCTURE  api_error, sandbox_error, unknown_api_state
#   MODEL'S FAULT   wrong_answer, compile_error, compile_timeout,
#                   runtime_error, timeout
#
# That channel is then DISCARDED one layer up:
#
#   verl/utils/reward_score/__init__.py:114
#     return float(res[0])          # <-- metadata dropped on the floor
#
# So `score = passed_count / total_cases` counts a test case that failed because
# the sandbox was DOWN exactly like one the model got wrong, and nothing raises.
# The deflation is completely silent, which is why `Exception in
# default_compute_score` has always read 0 while CODE was being biased downward.
#
# Measured impact on the r=32 control run: 468 exhausted-retry API errors.
# In the r=32 run the error density per validation window was 10.7 / 6.9 / 27.8
# per 1k log lines at steps 50 / 100 / 150 — a 4x spike in the window whose CODE
# score was lowest. A drifting scorer manufactures apparent capability decay.
#
# THE FIX, in three parts:
#   1. Call sandbox_fusion.compute_score DIRECTLY so metadata survives.
#      `continuous=True` is passed to match default_compute_score exactly, so
#      scores stay numerically comparable to every previous run.
#   2. RETRY when an infrastructure status appears. A transient sandbox outage
#      then yields a CORRECT score instead of a silent zero. Genuine failures
#      (wrong_answer / compile_error / runtime_error / timeout) are NOT retried.
#   3. Count every status and emit a periodic summary, so the infra-failure rate
#      is a LOGGED NUMBER per data_source instead of something inferred by
#      grepping the driver log.
#
# Any unexpected shape from the verl internals falls back to
# default_compute_score, so a verl upgrade degrades to old behaviour rather than
# crashing training.
# -----------------------------------------------------------------------------
_INFRA_STATUSES = frozenset({"api_error", "sandbox_error", "unknown_api_state"})

# Retries applied ONLY to infrastructure failures.
_SANDBOX_INFRA_RETRIES = int(os.getenv("SANDBOX_INFRA_RETRIES", "2"))
_SANDBOX_INFRA_BACKOFF = float(os.getenv("SANDBOX_INFRA_BACKOFF", "2.0"))

# Emit a status summary every N code samples (0 disables).
_SANDBOX_STATS_EVERY = int(os.getenv("SANDBOX_STATS_EVERY", "500"))

_stats_lock = threading.Lock()
_status_counts: dict[str, dict[str, int]] = {}
_code_samples_scored = 0
_infra_recovered = 0
_infra_unrecovered = 0


def _record_statuses(data_source: str, statuses: list, recovered: bool, had_infra: bool) -> None:
    global _code_samples_scored, _infra_recovered, _infra_unrecovered
    with _stats_lock:
        _code_samples_scored += 1
        bucket = _status_counts.setdefault(data_source, {})
        for st in statuses:
            bucket[str(st)] = bucket.get(str(st), 0) + 1
        if had_infra:
            if recovered:
                _infra_recovered += 1
            else:
                _infra_unrecovered += 1
        due = _SANDBOX_STATS_EVERY > 0 and _code_samples_scored % _SANDBOX_STATS_EVERY == 0
        if not due:
            return
        total = _code_samples_scored
        rec, unrec = _infra_recovered, _infra_unrecovered
        snapshot = {k: dict(v) for k, v in _status_counts.items()}
    # Log outside the lock.
    logger.warning(
        "[SANDBOX-STATS] code_samples=%d infra_failures: recovered=%d (%.2f%%) "
        "UNRECOVERED=%d (%.2f%%) <- these are scored as wrong answers | per_source=%s",
        total, rec, 100.0 * rec / max(total, 1), unrec, 100.0 * unrec / max(total, 1), snapshot,
    )


def _score_code_with_metadata(url, memory_limit_mb, completion, test_cases, data_source):
    """Score a code sample via verl's sandbox scorer, preserving metadata.

    Returns the float score. Raises if verl's internals do not behave as expected.

    Deliberately does NOT degrade to default_compute_score. scripts/runtime_env.yaml pins
    verl to exactly v0.8.0, so there is no version to be compatible with, and the
    fallback is the scorer this module's header spends 45 lines explaining is broken --
    it drops the metadata that distinguishes "the sandbox was down" from "the model got
    it wrong". A silent return to that path produces neither an error nor a
    [SANDBOX-UNRECOVERED] line, so the operator sees a deflated score and no signal.
    Failing loudly at submit time is the correct trade.
    """
    last_score = None
    saw_infra_earlier = False
    for attempt in range(_SANDBOX_INFRA_RETRIES + 1):
        res = sandbox_fusion.compute_score(
            url,
            _sandbox_semaphore,
            memory_limit_mb,
            completion,
            test_cases,
            continuous=True,  # MUST match default_compute_score for score parity
        )
        # Expected: (score, [metadata, ...]). Raise rather than returning None: falling
        # back to default_compute_score would silently reinstate the metadata-dropping
        # bug this function exists to fix, and a verl upgrade that changes this shape is
        # something the operator must see, not something to route around.
        if not (isinstance(res, tuple) and len(res) == 2):
            raise TypeError(
                f"sandbox_fusion.compute_score returned {type(res).__name__} "
                f"{res!r:.120}, expected a 2-tuple of (score, metadata_list). verl's "
                f"return shape changed; scripts/runtime_env.yaml pins v0.8.0."
            )
        score, meta = res
        if not isinstance(meta, list):
            raise TypeError(
                f"sandbox_fusion.compute_score returned metadata of type "
                f"{type(meta).__name__}, expected list. Infra-vs-wrong-answer "
                f"classification depends on the per-test-case status entries."
            )

        statuses = [m.get("status") for m in meta if isinstance(m, dict)]
        had_infra = any(s in _INFRA_STATUSES for s in statuses)
        last_score = float(score)

        if not had_infra:
            # Clean result. If an earlier attempt had failed on infrastructure,
            # this row was RECOVERED -- it would have been a silent zero before.
            _record_statuses(data_source, statuses, recovered=saw_infra_earlier,
                             had_infra=saw_infra_earlier)
            if saw_infra_earlier:
                logger.warning(
                    "[SANDBOX-RECOVERED] %s: retry succeeded, score %.3f "
                    "(previously would have been silently deflated)",
                    data_source, last_score,
                )
            return last_score

        saw_infra_earlier = True
        if attempt < _SANDBOX_INFRA_RETRIES:
            logger.warning(
                "[SANDBOX-RETRY] %s: infrastructure failure %s on attempt %d/%d, "
                "retrying in %.1fs (score would otherwise be a silent %.3f)",
                data_source,
                sorted({s for s in statuses if s in _INFRA_STATUSES}),
                attempt + 1,
                _SANDBOX_INFRA_RETRIES + 1,
                _SANDBOX_INFRA_BACKOFF * (attempt + 1),
                last_score,
            )
            time.sleep(_SANDBOX_INFRA_BACKOFF * (attempt + 1))
            continue

        # Retries exhausted: this row IS mis-scored. Count it loudly.
        _record_statuses(data_source, statuses, recovered=False, had_infra=True)
        logger.warning(
            "[SANDBOX-UNRECOVERED] %s: sandbox unreachable after %d attempts; "
            "score %.3f is a MEASUREMENT ARTIFACT, not model capability",
            data_source, _SANDBOX_INFRA_RETRIES + 1, last_score,
        )
        return last_score

    return last_score


def _log_sample(data_source, route, effective_source, result, ground_truth, solution_str):
    """Diagnostic logging for the first N samples per data_source."""
    count = _log_counts.get(data_source, 0)
    if count >= _MAX_LOG_SAMPLES:
        return
    _log_counts[data_source] = count + 1
    score_val = result["score"] if isinstance(result, dict) else result
    logger.warning(
        "[REWARD] sample=%d data_source=%s route=%s effective=%s "
        "score=%s gt_preview=%.80s response_tail=%.120s",
        count,
        data_source,
        route,
        effective_source,
        score_val,
        str(ground_truth),
        solution_str[-120:] if solution_str else "",
    )


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    memory_limit_mb=1024,
    **kwargs,
):
    """Wrapper around verl's default_compute_score with sandbox URL injection
    and defensive fallback for unrecognized data_source values.

    All parameters are passed straight through to default_compute_score when
    the data_source is natively supported.  For unknown data_source values
    (e.g. "eurus", "PRIME-RL/Eurus-2-RL-Data"), the function infers whether
    the sample is a math or code task from the ground_truth format and
    re-routes to the appropriate handler.

    The **kwargs catch any additional arguments verl main may pass (e.g.,
    reward_router_address from the Reward Loop, concurrent_semaphore, etc.).
    """
    # Determine the effective data_source for routing.
    # A val-holdout label (e.g. "numina_cn_k12_h2") resolves to its base source
    # here, so it is scored by exactly the same path as the legacy rows it will be
    # compared against, while `data_source` below keeps the original label for
    # verl's metric key and the [REWARD] logger.
    effective_source, route = _resolve_route(data_source, ground_truth)

    # Call verl's default scoring
    # concurrent_semaphore bounds in-flight sandbox execute calls; without it
    # verl runs unbounded and OOM-kills the sandbox pods (see module header).
    # Respect an explicit semaphore from verl if one is ever supplied.
    kwargs.setdefault("concurrent_semaphore", _sandbox_semaphore)

    # Sandbox-failure path: for CODE tasks, score via the sandbox directly so the metadata
    # (and therefore api_error vs wrong_answer) survives, and infrastructure
    # failures get retried instead of silently becoming zeros. Falls back to
    # default_compute_score if the verl internals are not the expected shape.
    is_code = route == "fallback-code" or effective_source in _CODE_DATA_SOURCES
    if is_code and sandbox_fusion_url:
        try:
            direct = _score_code_with_metadata(
                sandbox_fusion_url, memory_limit_mb, solution_str, ground_truth, data_source
            )
        except TypeError:
            # verl's return contract changed. That is a version problem, not a transient
            # failure, and falling back here would silently reinstate the scorer that
            # cannot tell "sandbox was down" from "model was wrong" -- for the whole run.
            # Propagate so it surfaces immediately instead of as a deflated CODE number.
            raise
        except Exception:  # noqa: BLE001 — a transient sandbox failure must not kill a 45h run
            logger.exception(
                "[REWARD] direct sandbox scoring failed (data_source=%s), "
                "falling back to default_compute_score",
                data_source,
            )
            direct = None
        if direct is not None:
            result = direct
            _log_sample(data_source, route, effective_source, result, ground_truth, solution_str)
            return result

    try:
        result = default_compute_score(
            data_source=effective_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            sandbox_fusion_url=sandbox_fusion_url,
            memory_limit_mb=memory_limit_mb,
            **kwargs,
        )
    except Exception:
        logger.exception(
            "[REWARD] Exception in default_compute_score "
            "(data_source=%s, effective=%s, route=%s, gt_preview=%s)",
            data_source,
            effective_source,
            route,
            str(ground_truth)[:80],
        )
        raise

    _log_sample(data_source, route, effective_source, result, ground_truth, solution_str)
    return result
