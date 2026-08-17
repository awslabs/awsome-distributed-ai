#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Offline tests for reward routing of val-holdout data_source labels.

No GPU, no Ray, no /fsx, no network. `custom_reward_fn` imports verl at module
level, so verl is stubbed in sys.modules before import -- the routing decision
itself is pure and needs none of it.

WHAT THIS PROTECTS
The MATH val aggregate has n=515 and a +-0.060 margin, the same size as the
effect it is used to measure. Enlarging it means appending never-trained rows,
and those rows need a data_source that verl reports separately, because verl
keys metrics as `val-core/<data_source>/acc/mean@1` and the legacy 515-row
aggregate must stay computable. Hence `numina_cn_k12` (legacy) vs
`numina_cn_k12_h2` (holdout).

The hazard: a suffixed label is not in _KNOWN_DATA_SOURCES, so without the
suffix strip it takes the `fallback-math` path and is scored as
`numina_olympiads` rather than as `numina_cn_k12`. Both land in prime_math
today, so the scores would probably agree -- but "probably agree" is not a
property you can put in a results table. If they ever diverged, the legacy and
holdout aggregates would become incomparable SILENTLY, and the comparison they
exist to support is the one carrying the project's only significant result.

So the requirement under test is exact: a holdout label must resolve to the
IDENTICAL (effective_source, route) as its legacy twin, for every source, and
nothing else about routing may change.

Run: python3 scripts/test_reward_routing.py
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

# --- stub verl so custom_reward_fn imports without the real package ----------
_verl = types.ModuleType("verl")
_utils = types.ModuleType("verl.utils")
_rs = types.ModuleType("verl.utils.reward_score")
_rs.default_compute_score = lambda **kw: 0.0  # never called by these tests
_sbf = types.ModuleType("verl.utils.reward_score.sandbox_fusion")
_sbf.compute_score = lambda *a, **kw: (0.0, [])
_sbf_utils = types.ModuleType("verl.utils.reward_score.sandbox_fusion.utils")
sys.modules.setdefault("verl", _verl)
sys.modules.setdefault("verl.utils", _utils)
sys.modules.setdefault("verl.utils.reward_score", _rs)
sys.modules.setdefault("verl.utils.reward_score.sandbox_fusion", _sbf)
sys.modules.setdefault("verl.utils.reward_score.sandbox_fusion.utils", _sbf_utils)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import custom_reward_fn as crf  # noqa: E402

_results: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail and not ok else ''}")


# Representative ground truths. Math answers are bare scalars/strings; code
# ground truth is a JSON object carrying test cases.
MATH_GT = "42"
MATH_GT_LATEX = "\\frac{1}{2}"
CODE_GT = json.dumps({"inputs": ["1 2\n"], "outputs": ["3\n"]})

LEGACY_MATH = [
    "numina_cn_k12",
    "numina_synthetic_math",
    "numina_olympiads",
    "numina_synthetic_amc",
    "numina_aops_forum",
    "numina_amc_aime",
]
LEGACY_CODE = ["apps", "codecontests", "codeforces", "taco"]

print("=== reward routing: holdout labels must score identically to legacy ===\n")

# 1. THE CORE PROPERTY. For every legacy source, the _h2 twin must resolve to the
#    exact same (effective_source, route). This is the whole point.
print("1 holdout twin resolves identically to its legacy source")
for src in LEGACY_MATH:
    want = crf._resolve_route(src, MATH_GT)
    got = crf._resolve_route(f"{src}{crf._HOLDOUT_SUFFIX}", MATH_GT)
    check(f"1 {src}_h2 -> {got} == legacy {want}", got == want, f"got {got} want {want}")
for src in LEGACY_CODE:
    want = crf._resolve_route(src, CODE_GT)
    got = crf._resolve_route(f"{src}{crf._HOLDOUT_SUFFIX}", CODE_GT)
    check(f"1 {src}_h2 -> {got} == legacy {want}", got == want, f"got {got} want {want}")

# 2. NEGATIVE CONTROL. Without the strip, a suffixed math label would land on
#    numina_olympiads via fallback-math. Prove the suffix strip is what prevents
#    that, so this suite cannot pass vacuously.
print("\n2 negative control: the strip is load-bearing")
eff_no_strip, route_no_strip = (
    ("numina_olympiads", "fallback-math")
    if "numina_cn_k12_h2" not in crf._KNOWN_DATA_SOURCES
    else ("?", "?")
)
got = crf._resolve_route("numina_cn_k12_h2", MATH_GT)
check(
    "2 numina_cn_k12_h2 is NOT scored as numina_olympiads/fallback-math",
    got != (eff_no_strip, route_no_strip),
    f"got {got}",
)
check(
    "2b suffixed label is genuinely absent from _KNOWN_DATA_SOURCES "
    "(so native routing comes from the strip, not from the set)",
    "numina_cn_k12_h2" not in crf._KNOWN_DATA_SOURCES,
)

# 3. REGRESSION GUARD. Every pre-existing routing decision must be untouched.
print("\n3 regression guard: legacy routing unchanged")
for src in LEGACY_MATH:
    check(f"3 {src} routes native to itself", crf._resolve_route(src, MATH_GT) == (src, "native"))
for src in LEGACY_CODE:
    check(f"3 {src} routes native to itself", crf._resolve_route(src, CODE_GT) == (src, "native"))
check(
    "3 unknown source + math gt still falls back to prime_math",
    crf._resolve_route("PRIME-RL/Eurus-2-RL-Data", MATH_GT) == ("numina_olympiads", "fallback-math"),
)
check(
    "3 unknown source + code gt still falls back to sandbox",
    crf._resolve_route("some_new_code_set", CODE_GT) == ("apps", "fallback-code"),
)
check("3 aime* prefix path preserved", crf._resolve_route("aime2024", MATH_GT) == ("aime2024", "native"))

# 4. CONSERVATISM. The strip must only fire when the base name is a real source,
#    so an unrelated name ending in _h2 cannot be silently rewritten.
print("\n4 strip is conservative")
check(
    "4 unrelated *_h2 name is NOT stripped into something else",
    crf._routing_source("totally_unrelated_h2") == "totally_unrelated_h2",
)
check("4b known base IS stripped", crf._routing_source("taco_h2") == "taco")
check("4c non-suffixed name untouched", crf._routing_source("taco") == "taco")
check(
    "4d unrelated *_h2 with math gt still reaches the math fallback",
    crf._resolve_route("totally_unrelated_h2", MATH_GT) == ("numina_olympiads", "fallback-math"),
)

# 5. Holdout CODE labels must still be recognised as code, or bug #8's metadata
#    path (and the sandbox semaphore) would be bypassed for them.
print("\n5 holdout code labels stay on the sandbox path")
for src in LEGACY_CODE:
    eff, _ = crf._resolve_route(f"{src}{crf._HOLDOUT_SUFFIX}", CODE_GT)
    check(f"5 {src}_h2 effective source is in _CODE_DATA_SOURCES", eff in crf._CODE_DATA_SOURCES)

# 6. Ground-truth shape detection must not be perturbed by the label.
print("\n6 ground-truth detection unaffected by label")
check("6 math scalar not detected as code", not crf._is_code_ground_truth(MATH_GT))
check("6 latex answer not detected as code", not crf._is_code_ground_truth(MATH_GT_LATEX))
check("6 code json detected as code", crf._is_code_ground_truth(CODE_GT))
check("6 dict form detected as code", crf._is_code_ground_truth({"inputs": [], "outputs": []}))
check(
    "6 holdout math label + latex gt still native",
    crf._resolve_route("numina_olympiads_h2", MATH_GT_LATEX) == ("numina_olympiads", "native"),
)

# 7. Defensive: non-string labels must not raise (verl has handed us surprises).
print("\n7 malformed labels do not raise")
for bad in (None, 123, b"taco_h2"):
    try:
        crf._routing_source(bad)
        check(f"7 _routing_source({bad!r}) handled", True)
    except Exception as exc:  # noqa: BLE001
        check(f"7 _routing_source({bad!r}) handled", False, repr(exc))

print()
failed = [n for n, ok in _results if not ok]
print(f"{len(_results) - len(failed)}/{len(_results)} checks passed")
if failed:
    print("FAILED:")
    for n in failed:
        print(f"  - {n}")
    sys.exit(1)
print("ALL GREEN")
