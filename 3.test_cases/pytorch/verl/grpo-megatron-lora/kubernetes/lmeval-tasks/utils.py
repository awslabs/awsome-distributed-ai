# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Custom lm-eval task utils for pass@10 (HumanEval + MBPP) — Qwen3-235B benchmark.

Provides a parameterized `pass_at_k` (k from the task YAML) plus per-benchmark
prediction builders, so both humaneval_p10 and mbpp_p10 can report pass@1 AND
pass@10. Stock lm-eval hardcodes k=[1] (humaneval utils.pass_at_k is parameterized
but MBPP's utils.pass_at_1 is not), so this file is mounted via --include_path.

MBPP fewshot samples are reused from the installed lm-eval package to avoid
duplicating the long hardcoded list.

WHY THESE BUILDERS EXTRACT FENCED CODE (instrument bug #10)
-----------------------------------------------------------
`code_eval` executes `prediction + "\\n" + test_code`, so a prediction MUST be a
syntactically valid program that defines the target function.

The previous builders assumed COMPLETION mode:

    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]

That is correct only while `r` is a bare continuation of the function body. It
became wrong the moment the harness was corrected for a reasoning model
(`--apply_chat_template`, `until=["<|im_end|>"]`, `max_gen_toks=24576`): Qwen3
then replies with prose plus a ```python fence, so `doc["prompt"] + r` yields

    def has_close_elements(...):        <- prompt, body-less
        \"\"\"...\"\"\"
    Okay, I need to compare each pair   <- prose => SyntaxError

i.e. a SyntaxError on essentially every sample, scoring pass@1 ~= 0 while
returning HTTP 200. A silent zero, not a visible failure.

It also explains the retired 0.2634 HumanEval baseline: under the old
completion-style `until` stops, generation was cut before the prose, so a
minority of samples happened to concatenate into valid code.

The builders below therefore try, in order:
  1. the LAST fenced block that defines the target function -> use standalone
     (prepending the prompt would double-define it body-less => IndentationError)
  2. any fenced block treated as a bare body -> prompt + block
  3. no fence but the raw text defines the function -> slice from its def,
     carrying any contiguous top-level imports that precede it
  4. `prompt + r` — byte-identical to the legacy behaviour, so completion mode
     (and the documented chat-template revert) keeps working unchanged

Every candidate is passed through `_repair`, which drops trailing lines until the
program parses. That removes trailing commentary ("This works because...") which
would otherwise invalidate an answer that was actually correct.
"""

import ast
import re

import evaluate as hf_evaluate

_code_eval = hf_evaluate.load("code_eval")

# ```python / ```py / ```  ... closing ``` optional so a generation truncated at
# max_gen_toks still yields its (partial) block.
_FENCE_RE = re.compile(r"```(?:[Pp]ython3?|[Pp]y)?[ \t]*\r?\n(.*?)(?:```|\Z)", re.DOTALL)

# Bound the repair scan so a pathological generation cannot dominate scoring time.
_MAX_TRIM_LINES = 400

_IMPORT_RE = re.compile(r"^\s*(?:import\s|from\s+\S+\s+import\s)")


def pass_at_k(references, predictions, k=None):
    assert k is not None, "k must be set in the task YAML metric_list"
    if isinstance(k, int):
        k = [k]
    res = _code_eval.compute(references=references, predictions=predictions, k=k)
    return res[0]


# ---- extraction helpers ----
def _parses(src):
    try:
        ast.parse(src)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    return True


def _repair(src):
    """Return `src`, or the longest leading prefix of it that parses, else None.

    Trailing prose is the common failure: a reasoning model appends an
    explanation after the code. Dropping whole trailing lines recovers the
    program without guessing at its content.
    """
    if not src or not src.strip():
        return None
    if _parses(src):
        return src
    lines = src.split("\n")
    for _ in range(min(_MAX_TRIM_LINES, len(lines) - 1)):
        lines.pop()
        candidate = "\n".join(lines)
        if candidate.strip() and _parses(candidate):
            return candidate
    return None


def _fenced_blocks(text):
    """Fenced code blocks in document order (non-empty only)."""
    return [m.group(1) for m in _FENCE_RE.finditer(text) if m.group(1).strip()]


def _slice_from_def(text, needle):
    """Slice from the line defining `needle`, keeping contiguous imports above it."""
    idx = text.find(needle)
    if idx < 0:
        return None
    lines = text[: idx + 1].split("\n")
    start_line = len(lines) - 1  # 0-based index of the def line
    all_lines = text.split("\n")
    first = start_line
    # Walk back over blank lines and top-level imports so the program keeps them.
    probe = start_line - 1
    while probe >= 0:
        stripped = all_lines[probe].strip()
        if not stripped:
            probe -= 1
            continue
        if _IMPORT_RE.match(all_lines[probe]):
            first = probe
            probe -= 1
            continue
        break
    return "\n".join(all_lines[first:])


# ---- HumanEval ----
def build_predictions(resps, docs):
    out = []
    for resp, doc in zip(resps, docs, strict=False):
        prompt = doc["prompt"]
        entry = doc.get("entry_point") or ""
        out.append([_humaneval_candidate(r, prompt, entry) for r in resp])
    return out


def _humaneval_candidate(r, prompt, entry):
    needle = f"def {entry}" if entry else "def "
    blocks = _fenced_blocks(r)

    # 1. last fence that actually defines the target function -> standalone
    for block in reversed(blocks):
        if needle in block:
            fixed = _repair(block)
            if fixed:
                return fixed

    # 2. a fence that does not define it is a bare body -> needs the signature
    for block in reversed(blocks):
        fixed = _repair(prompt + block)
        if fixed:
            return fixed

    # 3. unfenced answer that nonetheless defines the function
    if needle in r:
        sliced = _slice_from_def(r, needle)
        if sliced:
            fixed = _repair(sliced)
            if fixed:
                return fixed

    # 4. legacy completion-mode behaviour, byte-identical
    return prompt + r


# ---- MBPP ----
def build_predictions_mbpp(resps, docs):  # noqa: ARG001 - docs unused; signature fixed by lm-eval
    return [[_mbpp_candidate(r) for r in resp] for resp in resps]


def _mbpp_candidate(r):
    # `until: ["[DONE]"]` normally strips this server-side; belt-and-braces.
    text = r.split("[DONE]")[0] if "[DONE]" in r else r

    # MBPP predictions are always standalone programs (never prompt-prepended),
    # so a fence, when present, is the whole answer.
    for block in reversed(_fenced_blocks(text)):
        fixed = _repair(block)
        if fixed:
            return fixed

    # Legacy: 3-shot completion output is already plain code.
    fixed = _repair(text)
    if fixed:
        return fixed
    return text


def list_fewshot_samples():
    # Reuse the stock MBPP fewshot list from the installed package.
    from lm_eval.tasks.mbpp.utils import list_fewshot_samples as _lfs

    return _lfs()


# ---- BigCodeBench (Complete split) ----
# BigCodeBench ships, per task:
#   complete_prompt   — imports + `def task_func(...):` + PEP257 docstring
#   canonical_solution— ground-truth body
#   test              — a `unittest.TestCase` subclass (named `TestCases`) exercising task_func
#   entry_point       — always "task_func"
# code_eval runs `prediction + "\n" + reference` and marks pass iff it exits 0.
# The prediction must be a standalone program defining task_func; the reference is
# the unittest class plus a runner that RAISES on any failure/error (so a non-zero
# exit, which is what code_eval detects).

_BCB_ENTRY = "task_func"


def build_predictions_bcb(resps, docs):
    out = []
    for resp, doc in zip(resps, docs, strict=False):
        prompt = doc["complete_prompt"]
        out.append([_bcb_candidate(r, prompt) for r in resp])
    return out


def _bcb_candidate(r, prompt):
    needle = f"def {_BCB_ENTRY}"
    blocks = _fenced_blocks(r)

    # 1. last fence that defines task_func -> standalone program
    for block in reversed(blocks):
        if needle in block:
            fixed = _repair(block)
            if fixed:
                return fixed

    # 2. a fence that does not define it is a bare body -> prepend the prompt
    #    (imports + signature + docstring), same as HumanEval path 2
    for block in reversed(blocks):
        fixed = _repair(prompt + block)
        if fixed:
            return fixed

    # 3. unfenced answer that defines task_func -> slice from its def (+imports)
    if needle in r:
        sliced = _slice_from_def(r, needle)
        if sliced:
            fixed = _repair(sliced)
            if fixed:
                return fixed

    # 4. legacy completion-mode: prompt + raw continuation
    return prompt + r


def reference_bcb(doc):
    """Reference program appended after the prediction by code_eval.

    Runs the task's unittest.TestCase (BigCodeBench names it `TestCases`) and
    raises on any failure/error, which code_eval detects as a failed sample.

    Loads tests from the class object directly out of the executing globals
    rather than via loadTestsFromModule(sys.modules[__name__]): under code_eval
    the program is exec'd into a fresh globals dict that is NOT registered in
    sys.modules, so module-based discovery finds an EMPTY suite and a wrong
    program would score as passing. `loadTestsFromTestCase(TestCases)` binds to
    the actual class and is what surfaced 'NoneType is not callable' in the gate
    when discovery failed.
    """
    test_src = doc["test"]
    runner = (
        "\n\nimport unittest as _unittest\n"
        "_tc = globals().get('TestCases')\n"
        "assert _tc is not None, 'bigcodebench: TestCases class not defined'\n"
        "_suite = _unittest.TestLoader().loadTestsFromTestCase(_tc)\n"
        "assert _suite.countTestCases() > 0, 'bigcodebench: no test cases found'\n"
        "_res = _unittest.TextTestRunner(verbosity=0).run(_suite)\n"
        "assert _res.wasSuccessful(), 'bigcodebench tests failed'\n"
    )
    return test_src + runner
