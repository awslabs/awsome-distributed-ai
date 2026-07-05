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


# --- wall: Megatron validate_args probes the GPU on a GPU-less Ray driver -----
# Megatron's validate_args eagerly probes the CUDA device during pure argument
# validation (which SLIME runs on the Ray driver -- intentionally GPU-less in
# this test case: the head is num-gpus:0, GCS/dashboard only). Two probes are
# only reached by MoE models with tensor/context parallelism (the 30B recipe:
# --moe-grouped-gemm, TP=2, CP=2), so the 4B path never hits them:
#   * torch.cuda.get_device_capability()      (moe_grouped_gemm compute-cap assert)
#   * megatron.training.utils.get_device_arch_version() (imported by name into
#     megatron.training.arguments; used for the CUDA_DEVICE_MAX_CONNECTIONS note)
# On the GPU-less driver both raise "Found no NVIDIA driver". The real GPU actors
# (torch.cuda.is_available() == True) probe the real device unchanged.
#
# This injects a guard at the top of SLIME's own validate_args wrapper
# (slime/backends/megatron_utils/arguments.py) that, ONLY when no CUDA device is
# present (i.e. the driver), makes those two probes return safe values so
# argument validation can complete:
#   * get_device_capability -> (8, 0): satisfies the moe_grouped_gemm assert
#     (dc[0] >= 8) without a device; does not affect any arch-dependent branch.
#   * get_device_arch_version -> a sentinel (9999) that is NOT any real GPU
#     generation (Ampere=8, Hopper=9, Blackwell=10, ...), so it never mislabels
#     the hardware -- on a GPU-less driver the arch is genuinely unknown. Being
#     >= 10, it makes the driver SKIP the arch<10 CUDA_DEVICE_MAX_CONNECTIONS
#     branch, deferring that decision to the real GPU actors. Returning a real
#     generation (9 or 10) would falsely claim a specific arch; the sentinel does
#     not. The actual requirement is enforced on the actors from their real arch
#     (H200 = sm_90 requires it, satisfied via the recipe env; B300 = sm_100 does not).
# Guarded so substitution happens only while is_available() is False, and
# idempotent via a _slime_cpu_guard marker.
_VALIDATE_ANCHOR = '''def validate_args(args):
    """Run megatron\'s own validate_args plus slime-specific megatron validations."""
    _megatron_validate_args(args)'''

_VALIDATE_INJECT = '''def validate_args(args):
    """Run megatron\'s own validate_args plus slime-specific megatron validations."""
    # Megatron validate_args eagerly probes the CUDA device (get_device_capability
    # for moe_grouped_gemm; get_device_arch_version for the TP/CP
    # CUDA_DEVICE_MAX_CONNECTIONS note). SLIME runs validate_args on the Ray
    # driver, which is intentionally GPU-less in this test case, so those probes
    # raise "Found no NVIDIA driver". Guard them for the GPU-less driver only; the
    # real GPU actors probe the real device unchanged. Injected as a stopgap until
    # Megatron guards these probes with torch.cuda.is_available() upstream.
    import torch as _torch

    if not _torch.cuda.is_available():
        import megatron.training.arguments as _ma

        if not getattr(_torch.cuda.get_device_capability, "_slime_cpu_guard", False):
            _orig_cap = _torch.cuda.get_device_capability

            def _cap(*a, **k):
                return (8, 0) if not _torch.cuda.is_available() else _orig_cap(*a, **k)

            _cap._slime_cpu_guard = True
            _torch.cuda.get_device_capability = _cap

        if not getattr(_ma.get_device_arch_version, "_slime_cpu_guard", False):
            _orig_arch = _ma.get_device_arch_version
            # A GPU-less driver has no GPU arch to report. Return a value that is
            # deliberately NOT any real GPU generation (Ampere=8, Hopper=9,
            # Blackwell=10, ...) so we never mislabel the hardware; it just has to
            # be >= 10 so the arch-gated CUDA_DEVICE_MAX_CONNECTIONS branch is
            # skipped on the driver. The real requirement is decided on the GPU
            # actors from their real arch (they never re-run validate_args).
            _ARCH_UNKNOWN_ON_GPULESS_DRIVER = 9999

            def _arch(*a, **k):
                return (
                    _ARCH_UNKNOWN_ON_GPULESS_DRIVER
                    if not _torch.cuda.is_available()
                    else _orig_arch(*a, **k)
                )

            _arch._slime_cpu_guard = True
            _ma.get_device_arch_version = _arch

    _megatron_validate_args(args)'''


def _megatron_probe_is_unguarded() -> bool:
    """True if the installed Megatron's validate_args still probes the CUDA
    device eagerly without a torch.cuda.is_available() guard.

    This is the real defect this patch works around. When Megatron guards the
    probe upstream (the permanent fix), this returns False and the SLIME-side
    patch self-neutralizes. Locating Megatron via import keeps this robust to
    the install path. If Megatron cannot be located or its arguments module has
    been restructured beyond recognition, we conservatively assume the probe is
    still unguarded (better a harmless is_available()-gated guard than a crash).
    """
    try:
        import importlib.util

        spec = importlib.util.find_spec("megatron.training.arguments")
        if spec is None or not spec.origin:
            return True
        src = Path(spec.origin).read_text()
    except Exception:
        return True

    if "get_device_capability()" not in src and "get_device_arch_version()" not in src:
        # Probe removed/renamed entirely -> upstream changed it; nothing to guard.
        return False

    # A probe is considered guarded if a torch.cuda.is_available() check appears
    # on the probe's own line OR within the few lines immediately preceding it
    # (upstream's likely fix is `if ...is_available(): dc = get_device_capability()`
    # -- the guard sits on the enclosing `if`, not the probe line itself). If any
    # eager get_device_capability() call has no is_available() nearby, the probe
    # is still unguarded and the SLIME-side workaround is needed.
    lines = src.splitlines()
    WINDOW = 3
    for i, line in enumerate(lines):
        if "get_device_capability()" not in line:
            continue
        context = lines[max(0, i - WINDOW): i + 1]
        if not any("is_available" in c for c in context):
            return True
    return False


def patch_gpuless_driver_validate(slime_root: Path) -> PatchResult:
    """Fix: let Megatron validate_args run on the GPU-less Ray driver.

    Guards the two eager CUDA device probes (get_device_capability,
    get_device_arch_version) inside SLIME's validate_args wrapper so that on a
    GPU-less driver they return safe values instead of crashing with
    "Found no NVIDIA driver" (upstream Megatron issue/PR filed separately).
    """
    name = "gpuless-driver-validate"
    target = slime_root / "slime" / "backends" / "megatron_utils" / "arguments.py"
    if not target.is_file():
        return PatchResult(name, "error", f"target not found: {target}")

    src = target.read_text()

    if "_slime_cpu_guard" in src:
        return PatchResult(name, "already-applied", f"{target} already guards the driver probes")

    # True self-neutralization: the real defect is Megatron's *unguarded* eager
    # device probe in validate_args. If Megatron has guarded it upstream (the
    # permanent fix), this patch is unnecessary and must not touch SLIME. Detect
    # the unguarded probe in the installed Megatron; if it is gone, no-op.
    if not _megatron_probe_is_unguarded():
        return PatchResult(
            name,
            "already-fixed-upstream",
            "Megatron validate_args no longer has an unguarded CUDA device probe; leaving SLIME untouched",
        )

    # The probe is still unguarded upstream, so the SLIME-side guard is needed.
    # It must be injectable into SLIME's validate_args wrapper in its known shape.
    if _VALIDATE_ANCHOR not in src:
        return PatchResult(
            name,
            "error",
            "Megatron probe is unguarded but SLIME validate_args is not in the expected shape to inject a guard; "
            "refusing to edit (needs a refreshed anchor)",
        )
    if src.count(_VALIDATE_ANCHOR) != 1:
        return PatchResult(
            name,
            "error",
            f"expected exactly one validate_args wrapper, found {src.count(_VALIDATE_ANCHOR)}",
        )

    patched = src.replace(_VALIDATE_ANCHOR, _VALIDATE_INJECT, 1)
    try:
        compile(patched, str(target), "exec")
    except SyntaxError as exc:
        return PatchResult(name, "error", f"patched file fails to compile: {exc}")

    target.write_text(patched)
    return PatchResult(name, "applied", "guarded get_device_capability / get_device_arch_version on the GPU-less driver")


PATCHES = [
    patch_tms_preload_selection,
    patch_gpuless_driver_validate,
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
