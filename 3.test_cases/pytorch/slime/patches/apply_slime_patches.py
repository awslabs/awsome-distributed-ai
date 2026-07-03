#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Self-neutralizing patches for the pinned upstream SLIME checkout.

This test case installs SLIME straight from upstream (``THUDM/slime`` at the
pinned ``SLIME_VERSION`` -- no fork, no forked URL). A handful of upstream bugs
still block a clean end-to-end run on B300 / H200 (CUDA 13). Rather than fork
SLIME or hard-code a fork URL, this script applies each fix *in place* against
the upstream checkout, and only when the unfixed pattern is actually present.

Design goals (why this shape):
  * Default is upstream. The image installs upstream SLIME as-is; this runs
    afterwards as a thin, auditable layer.
  * Self-neutralizing. Every patch first checks whether the upstream code still
    exhibits the bug. Once upstream merges the corresponding fix, the pattern is
    gone and the patch becomes a no-op automatically -- nothing to remember, no
    version pin to bump, no fork to track. Upstream simply wins.
  * Idempotent. Re-running (or running against an already-patched tree) is safe;
    an applied patch is detected and skipped.
  * Minimal blast radius. Each patch is scoped to the smallest possible edit and
    is a no-op unless its exact precondition matches, so an unexpected upstream
    refactor makes the patch skip (and say so) rather than corrupt the file.

Each patch links to the upstream issue/PR that will make it unnecessary. When
all patches report "already fixed upstream", this file can be deleted.

Usage (from the Dockerfile, right after the upstream SLIME install):
    python3 patches/apply_slime_patches.py --slime-root /opt/slime

Exit code is 0 whenever every patch ends in a known-good state (applied,
already-applied, or already-fixed-upstream). It is non-zero only if a patch's
target file is missing or a patch is genuinely unable to reach a good state, so
a broken image fails the build loudly instead of silently shipping the bug.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# --- helper injected into slime/ray/actor_group.py --------------------------
# Identical in behavior to the upstream fix proposed in the accompanying PR:
# delegate .so selection to torch_memory_saver's own CUDA-aware resolver, with a
# loadability-based fallback for older torch_memory_saver builds that predate it.
_TMS_HELPER = '''

def _resolve_tms_preload_lib(torch_memory_saver):
    """Path to the torch_memory_saver preload .so that matches the CUDA runtime.

    Prefer the library's own CUDA-aware resolver. Fall back -- for older
    torch_memory_saver builds that lack it -- to a candidate list that includes
    the cu<major> variant for the *detected* CUDA major and picks the first that
    actually loads (existence is not loadability: a cu12 .so exists on a CUDA 13
    box but cannot be dlopen'd, which is what makes this fail on CUDA 13).

    Injected by awsome-distributed-training patches/apply_slime_patches.py as a
    stopgap until the equivalent fix lands upstream in THUDM/slime.
    """
    import os as _os

    stem = "torch_memory_saver_hook_mode_preload"

    try:
        from torch_memory_saver.utils import get_binary_path_from_package

        return str(get_binary_path_from_package(stem))
    except Exception:
        pass

    import ctypes

    base = _os.path.dirname(_os.path.dirname(torch_memory_saver.__file__))
    try:
        import torch

        cuda = getattr(torch.version, "cuda", None)
        major = cuda.split(".", 1)[0] if cuda else None
    except Exception:
        major = None

    candidates = []
    if major:
        candidates.append(f"{stem}_cu{major}.abi3.so")
    candidates += [f"{stem}.abi3.so", f"{stem}_cu12.abi3.so", f"{stem}_cu13.abi3.so"]

    tried = []
    for name in candidates:
        path = _os.path.join(base, name)
        if not _os.path.exists(path):
            continue
        try:
            ctypes.CDLL(path)
        except OSError as exc:
            tried.append(f"{name}: {exc}")
            continue
        return path

    raise FileNotFoundError(
        "Could not find a loadable torch_memory_saver preload library for the "
        f"current CUDA runtime under {base}. Tried: {tried or candidates}"
    )
'''

_HELPER_MARKER = "def _resolve_tms_preload_lib("

# The unfixed upstream pattern hard-codes the preload .so filename(s) and selects
# by existence. We match on the filename token that only appears in that unfixed
# path; once upstream selects by CUDA runtime, this token is gone and the patch
# self-neutralizes.
_BROKEN_TOKEN = '"torch_memory_saver_hook_mode_preload.abi3.so"'
_ENV_ASSIGN = 'env_vars["LD_PRELOAD"] = dynlib_path'
_ENV_ASSIGN_REPLACEMENT = (
    "dynlib_path = _resolve_tms_preload_lib(torch_memory_saver)\n\n"
    '            env_vars["LD_PRELOAD"] = dynlib_path'
)


class PatchResult:
    """Outcome of one patch: (status, message). status in known-good set => ok."""

    GOOD = {"applied", "already-applied", "already-fixed-upstream"}

    def __init__(self, name: str, status: str, message: str):
        self.name = name
        self.status = status
        self.message = message

    @property
    def ok(self) -> bool:
        return self.status in self.GOOD

    def __str__(self) -> str:
        flag = "OK " if self.ok else "ERR"
        return f"[{flag}] {self.name}: {self.status} -- {self.message}"


def patch_tms_preload_selection(slime_root: Path) -> PatchResult:
    """Fix: pick the torch_memory_saver LD_PRELOAD .so by CUDA runtime, not by
    filename existence (upstream issue/PR: torch_memory_saver cu13 selection).

    On CUDA 13, upstream selects a cu12-linked .so and every child dies with
    'libcudart.so.12: cannot open shared object file'. See the accompanying
    SLIME PR for the full analysis.
    """
    name = "tms-preload-cuda-aware"
    target = slime_root / "slime" / "ray" / "actor_group.py"
    if not target.is_file():
        return PatchResult(name, "error", f"target not found: {target}")

    src = target.read_text()

    # (a) already patched by us?
    if _HELPER_MARKER in src:
        return PatchResult(name, "already-applied", f"{target} already has the helper")

    # (b) upstream already fixed it? The unfixed path is identified by the
    # hard-coded preload filename token; if it is gone, there is nothing to fix.
    if _BROKEN_TOKEN not in src:
        return PatchResult(
            name,
            "already-fixed-upstream",
            "unfixed pattern absent; leaving upstream code untouched",
        )

    # (c) the assignment site we hang the helper call on must be present and
    # unique, or we refuse to edit (an unexpected refactor -> skip, do not guess).
    if src.count(_ENV_ASSIGN) != 1:
        return PatchResult(
            name,
            "error",
            f'expected exactly one occurrence of `{_ENV_ASSIGN}`, '
            f"found {src.count(_ENV_ASSIGN)}; refusing to edit",
        )

    # Insert the helper after the import block (after the last top-level import),
    # and route the existing assignment through it.
    lines = src.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith(("import ", "from ")):
            insert_at = i + 1
    patched = "".join(lines[:insert_at]) + _TMS_HELPER + "".join(lines[insert_at:])
    patched = patched.replace(_ENV_ASSIGN, _ENV_ASSIGN_REPLACEMENT, 1)

    # Byte-compile to guarantee we did not produce invalid Python.
    try:
        compile(patched, str(target), "exec")
    except SyntaxError as exc:
        return PatchResult(name, "error", f"patched file fails to compile: {exc}")

    target.write_text(patched)
    return PatchResult(name, "applied", f"routed LD_PRELOAD through {_HELPER_MARKER[:-1]}()")


PATCHES = [
    patch_tms_preload_selection,
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--slime-root",
        default="/opt/slime",
        type=Path,
        help="Path to the upstream SLIME checkout (default: /opt/slime).",
    )
    args = ap.parse_args()

    if not args.slime_root.is_dir():
        print(f"[patch] SLIME root not found: {args.slime_root}", file=sys.stderr)
        return 2

    print(f"[patch] applying self-neutralizing SLIME patches under {args.slime_root}")
    results = [patch(args.slime_root) for patch in PATCHES]
    for r in results:
        print(f"[patch] {r}")

    failed = [r for r in results if not r.ok]
    if failed:
        print(f"[patch] {len(failed)} patch(es) failed", file=sys.stderr)
        return 1
    print("[patch] all patches in a known-good state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
