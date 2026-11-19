#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Offline unit tests for kubernetes/lmeval-tasks/utils.py (fenced-code extraction).

Runs with NO GPU, NO network and NO vLLM. `utils.py` calls
`evaluate.load("code_eval")` at import time, so a stub `evaluate` module is
injected into sys.modules first; nothing here touches the real metric.

This file deliberately lives in scripts/ and NOT in kubernetes/lmeval-tasks/,
because submit_lmeval.sh ConfigMaps *every* file in that directory into /tasks.

Usage:  python3 scripts/test_lmeval_utils.py
"""

import importlib.util
import pathlib
import sys
import types

# ---- stub `evaluate` before importing utils.py ------------------------------
_stub = types.ModuleType("evaluate")
_stub.load = lambda *_a, **_k: types.SimpleNamespace(compute=lambda **_kw: ({}, {}))
sys.modules.setdefault("evaluate", _stub)

_UTILS = pathlib.Path(__file__).resolve().parents[1] / "kubernetes" / "lmeval-tasks" / "utils.py"
_spec = importlib.util.spec_from_file_location("lmeval_task_utils", _UTILS)
u = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(u)

# ---- fixtures --------------------------------------------------------------
HE_PROMPT = '''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer than threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    """
'''
ENTRY = "has_close_elements"
DOC = {"prompt": HE_PROMPT, "entry_point": ENTRY}

LEGACY_BODY = """    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
"""

FULL_FN = """from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
"""

_results = []


def check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))


def he(resp):
    """Single-response HumanEval build -> the one candidate string."""
    return u.build_predictions([[resp]], [DOC])[0][0]


def mb(resp):
    return u.build_predictions_mbpp([[resp]], [{}])[0][0]


print("=" * 78)
print("HumanEval builder")
print("=" * 78)

# 1. THE REGRESSION GUARD: completion mode must be byte-identical to legacy.
out = he(LEGACY_BODY)
check("1 no-fence completion output is byte-identical to legacy prompt+r", out == HE_PROMPT + LEGACY_BODY)
check("1b legacy result parses", u._parses(out))

# 2. The actual failure case: prose + fence defining the entry point.
chat = (
    "Okay, I need to check whether any two numbers in the list are closer\n"
    "together than the given threshold. Let me compare every pair.\n\n"
    "```python\n" + FULL_FN + "```\n\n"
    "This is O(n^2) but the inputs are small.\n"
)
out = he(chat)
check("2 prose+fence extracts the fence", "Okay, I need to check" not in out)
check("2b fence used standalone (entry not double-defined)", out.count(f"def {ENTRY}") == 1)
check("2c result parses", u._parses(out))
check("2d trailing prose removed", "O(n^2)" not in out)

# 3. Fence containing only a BODY -> must get the signature prepended.
body_fence = "```python\n" + LEGACY_BODY + "```\n"
out = he(body_fence)
check("3 body-only fence gets prompt prepended", out.startswith("from typing import List"))
check("3b result parses", u._parses(out))
check("3c defines the entry point exactly once", out.count(f"def {ENTRY}") == 1)

# 4. Multiple fences -> the LAST one that defines the entry point wins.
multi = (
    "First attempt:\n```python\ndef helper(x):\n    return x\n```\n"
    "That was wrong. Final answer:\n```python\n" + FULL_FN + "```\n"
)
out = he(multi)
check("4 last defining fence wins", "def helper" not in out)
check("4b result parses", u._parses(out))

# 5. Imports inside the fence survive.
out = he(chat)
check("5 imports preserved from fence", "from typing import List" in out)

# 6. Unterminated fence (generation truncated at max_gen_toks) still usable.
truncated = "Here is the solution:\n\n```python\n" + FULL_FN
out = he(truncated)
check("6 unterminated fence still extracted", "Here is the solution" not in out)
check("6b result parses", u._parses(out))

# 7. No fence, but the raw answer defines the function WITH a preceding import.
raw = "Sure. Let me write it.\n\n" + FULL_FN + "\nHope that helps!\n"
out = he(raw)
check("7 unfenced def is sliced out", "Sure. Let me write it." not in out)
check("7b preceding import carried along", "from typing import List" in out)
check("7c result parses", u._parses(out))
check("7d trailing prose removed", "Hope that helps" not in out)

# 8. Degenerate inputs must not raise.
for label, resp in [("empty", ""), ("whitespace", "   \n\n"), ("prose only", "I cannot solve this.")]:
    try:
        out = he(resp)
        check(f"8 {label} response handled without raising", True)
    except Exception as exc:  # noqa: BLE001
        check(f"8 {label} response handled without raising", False, repr(exc))

print()
print("=" * 78)
print("MBPP builder")
print("=" * 78)

MBPP_CODE = "def similar_elements(a, b):\n    return tuple(set(a) & set(b))\n"

# 9. Legacy 3-shot plain code -> byte-identical passthrough.
out = mb(MBPP_CODE)
check("9 plain code is byte-identical to legacy raw completion", out == MBPP_CODE)

# 10. Fenced answer -> extracted.
out = mb("Sure!\n\n```python\n" + MBPP_CODE + "```\n\nDone.")
check("10 fence extracted", out.strip().startswith("def similar_elements"))
check("10b result parses", u._parses(out))
check("10c prose removed", "Sure!" not in out and "Done." not in out)

# 11. [DONE] sentinel stripped if it leaks through the stop sequence.
out = mb(MBPP_CODE + "[DONE]\nsome trailing junk")
check("11 [DONE] and trailing junk stripped", "[DONE]" not in out and "junk" not in out)
check("11b result parses", u._parses(out))

# 12. Degenerate MBPP input.
try:
    mb("")
    check("12 empty MBPP response handled", True)
except Exception as exc:  # noqa: BLE001
    check("12 empty MBPP response handled", False, repr(exc))

print()
print("=" * 78)
print("PROPERTY: every non-legacy-fallback candidate must be syntactically valid")
print("=" * 78)
corpus = [chat, body_fence, multi, truncated, raw, "```python\n" + FULL_FN + "```"]
bad = [c for c in corpus if not u._parses(he(c))]
check("13 all realistic chat-mode responses yield parseable programs", not bad, f"{len(bad)} failed")

print()
print("=" * 78)
print("END-TO-END: candidate + real HumanEval test must EXECUTE and pass")
print("=" * 78)
# This is what code_eval does: exec(prediction + "\n" + test + f"check({entry})").
# Syntactic validity is necessary but not sufficient -- a body-less function
# parses fine and then returns None, silently scoring 0. This proves the
# extracted program is functionally correct, with no GPU and no vLLM.
HE_TEST = """

METADATA = {}


def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) is True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) is False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) is True
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) is True
"""


def executes(candidate):
    program = candidate + "\n" + HE_TEST + f"\ncheck({ENTRY})\n"
    try:
        exec(compile(program, "<candidate>", "exec"), {})  # noqa: S102
    except Exception:  # noqa: BLE001
        return False
    return True


check("14 legacy completion-mode candidate executes", executes(he(LEGACY_BODY)))
check("14b prose+fence candidate executes", executes(he(chat)))
check("14c body-only fence candidate executes", executes(he(body_fence)))
check("14d multi-fence candidate executes", executes(he(multi)))
check("14e truncated-fence candidate executes", executes(he(truncated)))
check("14f unfenced-with-prose candidate executes", executes(he(raw)))
# Negative control: the OLD builder on chat-mode output must FAIL, proving the
# test would have caught the extraction bug rather than passing vacuously.
check("14g NEGATIVE CONTROL: old prompt+r on chat output fails", not executes(HE_PROMPT + chat))

print()
failed = [n for n, ok, _ in _results if not ok]
total = len(_results)
print(f"{total - len(failed)}/{total} checks passed")
if failed:
    print("FAILED:")
    for n in failed:
        print(f"  - {n}")
    sys.exit(1)
print("ALL GREEN")
