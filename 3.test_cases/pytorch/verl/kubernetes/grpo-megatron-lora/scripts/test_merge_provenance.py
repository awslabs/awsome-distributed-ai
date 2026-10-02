#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Offline tests for the merge_adapter provenance gate (adapter-rank mismatch).

No GPU, no Ray, no /fsx. merge_adapter.py imports only stdlib at module level,
so it can be imported directly.

WHAT THIS PROTECTS
merge_adapter.sh defaults DATASET to `mixed-code-math-v2`, a tree that holds a
COMPLETE set of r=32/alpha=64 adapters including global_step_100. The r=128 run
writes elsewhere. `merge_adapter.sh 100` with defaults therefore merges the wrong
experiment's adapter -- and the non-zero gate, the delta gate and
check_merge_parity.py ALL pass, because none of them ask which adapter it is.
The result is the wrong model shipped under the right name with every check green.

Usage:  python3 scripts/test_merge_provenance.py
"""

import importlib.util
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent / "merge_adapter.py"
_spec = importlib.util.spec_from_file_location("merge_adapter_mod", _SRC)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

ADAPTER = pathlib.Path("/fsx/data/verl/ckpts/some-tree/global_step_100/actor/huggingface/adapter")
OURS = {"r": 128, "lora_alpha": 256}
OLD_EXPERIMENT = {"r": 32, "lora_alpha": 64}

_results = []


def check(name, cond, detail=""):
    _results.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))


def gate(cfg, rank, alpha):
    """Return (exited, stderr_text)."""
    import io

    err = io.StringIO()
    real = sys.stderr
    sys.stderr = err
    try:
        m._assert_adapter_provenance(cfg, rank, alpha, ADAPTER)
        return False, err.getvalue()
    except SystemExit:
        return True, err.getvalue()
    finally:
        sys.stderr = real


print("=" * 78)
print("merge_adapter provenance gate")
print("=" * 78)

# THE CASE THAT MATTERS: the real default-tree adapter against our expectation.
exited, msg = gate(OLD_EXPERIMENT, 128, 256)
check("1 r=32 adapter rejected when r=128 expected (the rank-mismatch scenario)", exited)
check("1b failure names the wrong-tree cause", "mixed-code-math-v2" in msg, msg[:200])
check("1c failure reports both values", "r=128" in msg and "r=32" in msg, msg[:200])

# Correct adapter must pass.
exited, _ = gate(OURS, 128, 256)
check("2 matching r=128/alpha=256 adapter passes", not exited)

# Alpha alone must be able to fail it: alpha/rank ratio is load-bearing for
# cross-run comparability, so r=128/alpha=64 is NOT an acceptable substitute.
exited, msg = gate({"r": 128, "lora_alpha": 64}, 128, 256)
check("3 correct rank but wrong alpha is rejected", exited)
check("3b failure mentions alpha", "ALPHA" in msg or "lora_alpha" in msg)

# Rank-only expectation still works (alpha unspecified).
exited, _ = gate(OURS, 128, None)
check("4 rank-only expectation passes on matching adapter", not exited)
exited, _ = gate(OLD_EXPERIMENT, 128, None)
check("4b rank-only expectation still catches r=32", exited)

# No expectation = permissive (backwards compatible), but must NOT claim to verify.
exited, _ = gate(OLD_EXPERIMENT, None, None)
check("5 no expectation is permissive (backwards compatible)", not exited)

# Missing keys must not crash the gate.
for label, cfg in [("empty cfg", {}), ("r only", {"r": 128}), ("null r", {"r": None})]:
    try:
        gate(cfg, 128, 256)
        check(f"6 {label} handled without raising", True)
    except Exception as exc:  # noqa: BLE001
        check(f"6 {label} handled without raising", False, repr(exc))

# A malformed expectation must fail closed, not open.
exited, _ = gate({"r": "128"}, 128, None)
check("7 string '128' != int 128 fails CLOSED (no silent coercion)", exited)

print()
failed = [n for n, ok in _results if not ok]
print(f"{len(_results) - len(failed)}/{len(_results)} checks passed")
if failed:
    print("FAILED:")
    for n in failed:
        print(f"  - {n}")
    sys.exit(1)
print("ALL GREEN")
