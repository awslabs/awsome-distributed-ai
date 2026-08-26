#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Opt-in draft-PR layer for the NeMo-RL + DeepEP V2 (EFA) test case.

The BASELINE image installs three upstream trees as-is (deepseek-ai/DeepEP,
NVIDIA/Megatron-LM, NVIDIA-NeMo/RL) at pinned SHAs and depends on nothing
else. The full GRPO rollout-over-DeepEP path additionally needs three upstream
PRs that are still DRAFT/open — this script bakes them in, and ONLY when the
image is built with ``--build-arg APPLY_DRAFT_ROLLOUT_PATCHES=1``. (A fourth,
NVIDIA-NeMo/RL#2411, is intentionally excluded — see the #2410 entry below.)

Design goals (why this shape — mirrors the slime sibling's patch layer):
  * Default is upstream. Running this script is opt-in; the baseline image
    never executes it, so it can never contaminate the upstream-only flavor.
  * Pinned, not floating. Each PR is applied as its individual commits at
    IMMUTABLE SHAs fetched from the upstream repo's own commit endpoint
    (``https://github.com/<org>/<repo>/commit/<sha>.patch``). A bare
    ``refs/pull/N/head`` is a moving ref and is never used.
  * Fail-loud. Every commit is ``git apply --check``ed before applying; if a
    hunk no longer applies against the pinned base, the BUILD fails — an image
    whose patch state is ambiguous must not ship.
  * Self-neutralizing, at CHANGE granularity. Each PR carries one content
    probe per independently-required change (not one probe for the whole PR).
    Before applying, ALL of a PR's probes are checked: only if EVERY probe
    already passes — the PR merged upstream and the tree pin advanced past all
    of it — is the PR skipped and reported. A partially-merged tree (some
    probes pass, some don't) is NOT skipped; its commits are applied, and a
    commit that then fails --check is a fail-loud build error (the pinned tree
    drifted), never a silent skip. When every probe reports already-present,
    delete the entry here. (A per-commit reverse-check is deliberately absent:
    it false-negatives on multi-commit PRs whose later commits touch the same
    hunks, and the all-probes decision above makes it redundant.)
  * Post-asserted, on the full list. After a PR applies, EVERY one of its
    probes is re-checked before the marker is written — so the marker can never
    record a patch state the tree does not actually have (a partially-present
    PR fails here), and verify-image.sh can trust the marker it reads.

Each entry links the upstream PR that will make it unnecessary.

Usage (from nemo-rl.Dockerfile, Layer 8):
    python3 apply_nemo_rl_patches.py \
        --deepep-root /opt/DeepEP --megatron-root /opt/Megatron-LM \
        --nemo-rl-root /opt/NeMo-RL --marker /opt/.draft-rollout-patches-applied
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# One entry per draft PR. `commits` are the PR's commits in order, at the
# immutable SHAs they had when this test case was authored (2026-08-25).
# `probes` = a list of (relative path, needle) — ONE per independently-required
# change in the PR (needle None = the file's existence is the assertion). The
# list is deliberately per-CHANGE, not per-PR: a single per-PR probe would let a
# partially-merged upstream tree satisfy the probe while lacking later commits,
# and the marker would then record a patch state the tree does not have. A PR is
# skipped only when EVERY probe passes; after applying, EVERY probe is
# re-asserted before the marker is written.
#
# Pure-cleanup commits (no detectable content addition — e.g. a comment/URL
# removal) intentionally have NO probe: commits are applied atomically in order
# and are never skipped individually, and a PR-level skip fires only when the
# whole PR is already upstream (i.e. merged past the cleanup too), so a cleanup
# commit can be verified by neither presence nor absence without ambiguity.
PATCH_SETS = [
    {
        "name": "deepseek-ai/DeepEP#612 — aws-efa: QP cap, get_rdma_gbs fast path, scaleout interval",
        "url": "https://github.com/deepseek-ai/DeepEP/pull/612",
        "repo": "deepseek-ai/DeepEP",
        "root_arg": "deepep_root",
        "commits": [
            "4eddba396006b8e4aa1a3a9a505396020aba4ef7",  # cap auto-QP at 2 on EFA (128-slot GIN ring)
            "922a1fa7c0cd3ef047c0919638a87f9a2360346b",  # EFA fast path in get_rdma_gbs (SM auto-sizing)
            "28d1f7fb173f728be51632ce0026fea23243e350",  # dispatch kScaleoutUpdateInterval 6 -> 16
        ],
        "probes": [
            ("deep_ep/buffers/elastic.py", "EP_EFA_MAX_QPS"),                       # commit 1
            ("deep_ep/utils/envs.py", "EP_EFA_RDMA_GBS"),                           # commit 2
            ("deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh",
             "kScaleoutUpdateInterval = 16"),                                        # commit 3
        ],
    },
    {
        "name": "NVIDIA/Megatron-LM#4632 — moe: DeepEP V2 ElasticBuffer support in the flex dispatcher",
        "url": "https://github.com/NVIDIA/Megatron-LM/pull/4632",
        "repo": "NVIDIA/Megatron-LM",
        "root_arg": "megatron_root",
        "commits": [
            "e132d5dd15358940aeb962105e44b402919084c5",  # ElasticBuffer support in _DeepepManager
            "f5ac3d481a8baca2596f20ae213dee25f87e35bf",  # graceful EventOverlap import fallback under V2
            "99b8824ee9c8d26b115e05b3ac563d1ea73b2b6b",  # pass num_experts explicitly to V2 backward dispatch
            "8056d6d489c73a353d590b8079497fceda4f9aa7",  # drop downstream repro URLs + dead conditional (cleanup — no probe)
        ],
        "probes": [
            ("megatron/core/transformer/moe/fused_a2a.py", "ElasticBuffer"),        # commit 1
            ("megatron/core/transformer/moe/fused_a2a.py", "deep_ep.utils.event"),  # commit 2
            ("megatron/core/transformer/moe/fused_a2a.py", "_handle_num_experts"),  # commit 3
        ],                                                                           # commit 4 = cleanup, see header
    },
    {
        "name": "NVIDIA-NeMo/RL#2410 — deps: re-export LD_LIBRARY_PATH for AWS EFA OFI discovery",
        "url": "https://github.com/NVIDIA-NeMo/RL/pull/2410",
        "repo": "NVIDIA-NeMo/RL",
        "root_arg": "nemo_rl_root",
        # This PR's commit is cut directly on NEMO_RL_SHA=46be4e8 (its parent),
        # so it applies --check-clean on the baseline tree. (#2411, the deep_ep
        # pin bump, is intentionally NOT applied here: its base is cc75cad —
        # 116 commits ahead of 46be4e8, across the requires-python 3.12->3.13.13
        # bump — so it neither applies here NOR belongs on this py3.12 substrate,
        # and it is metadata-only for this image since deep_ep is built from
        # /opt/DeepEP, not from NeMo-RL's pin.)
        "commits": [
            "7f0f21a7a8d7205d2d741f2bc9cff837462091a5",
        ],
        # The PR also ships the worked 2-node EFA GRPO recipe config this test
        # case's README points at for the full rollout path.
        "probes": [
            ("examples/configs/recipes/llm/aws-efa-grpo-qwen3-30ba3b-2n8g-megatron.yaml", None),
        ],
    },
]


def run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def fetch_patch(repo: str, sha: str) -> bytes:
    url = f"https://github.com/{repo}/commit/{sha}.patch"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    if not data.lstrip().startswith(b"From "):
        raise RuntimeError(f"{url} did not return a git patch")
    return data


def apply_commit(root: Path, repo: str, sha: str) -> str:
    """Apply one pinned commit into the tree at `root`. Returns a status word.

    No per-commit reverse-check: the whole-PR probe gate in main() decides
    skip-vs-apply (a PR is skipped only when EVERY probe already passes). Here we
    apply unconditionally and fail loud if the pinned tree drifted under a hunk —
    a per-commit reverse-check would false-negative on multi-commit PRs whose
    later commits touch the same hunks (the docstring's stated design).
    """
    # absolute: git apply runs with cwd=root, so a relative path would resolve
    # inside the tree twice
    patch_path = (root / f".{sha}.patch").resolve()
    patch_path.write_bytes(fetch_patch(repo, sha))
    try:
        check = run(["git", "apply", "--check", str(patch_path)], root)
        if check.returncode != 0:
            raise RuntimeError(
                f"{repo} commit {sha} does not apply cleanly:\n{check.stderr}\n"
                "The pinned tree moved under the patch — re-pin the tree SHA to the "
                "PR's current base, or retire this entry if the PR merged with changes."
            )
        applied = run(["git", "apply", str(patch_path)], root)
        if applied.returncode != 0:
            raise RuntimeError(f"{repo} commit {sha} --check passed but apply failed:\n{applied.stderr}")
        return "applied"
    finally:
        patch_path.unlink(missing_ok=True)


def probe_ok(root: Path, probe: tuple[str, str | None]) -> bool:
    rel, needle = probe
    target = root / rel
    if not target.is_file():
        return False
    return needle is None or needle in target.read_text(errors="replace")


def missing_probes(root: Path, probes: list[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
    """The subset of `probes` NOT yet satisfied in the tree at `root`."""
    return [p for p in probes if not probe_ok(root, p)]


def describe_probe(probe: tuple[str, str | None]) -> str:
    rel, needle = probe
    return f"{rel}" if needle is None else f"{needle!r} in {rel}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepep-root", required=True, type=Path)
    parser.add_argument("--megatron-root", required=True, type=Path)
    parser.add_argument("--nemo-rl-root", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path,
                        help="marker file recording the applied commit SHAs")
    args = parser.parse_args()
    roots = {
        "deepep_root": args.deepep_root,
        "megatron_root": args.megatron_root,
        "nemo_rl_root": args.nemo_rl_root,
    }

    record: list[str] = []
    for pset in PATCH_SETS:
        root = roots[pset["root_arg"]]
        if not (root / ".git").exists():
            print(f"FATAL: {root} is not a git checkout — cannot apply {pset['name']}", file=sys.stderr)
            return 1
        print(f"== {pset['name']} ==")
        probes = pset["probes"]
        missing = missing_probes(root, probes)
        if not missing:
            # Every per-change probe already passes — the tree pin advanced past
            # ALL of this PR's changes (fully merged upstream). Only then skip;
            # a partially-merged tree (some probes missing) is NOT skipped, so it
            # can never be recorded as already-present. Delete the entry here.
            print(f"   all {len(probes)} change-probes already present — skipping; retire this entry")
            record.extend(f"{pset['repo']}@{sha} already-present" for sha in pset["commits"])
            continue
        if len(missing) != len(probes):
            # Partially-present: some changes are in the tree, some are not. Apply
            # the PR's commits (git apply --check will fail loud on any hunk that
            # the already-present change collides with) rather than silently
            # trusting the single-needle hit the old per-PR probe would have.
            print(f"   partially present ({len(probes) - len(missing)}/{len(probes)} "
                  f"change-probes hit) — applying to complete: "
                  f"{', '.join(describe_probe(p) for p in missing)}")
        for sha in pset["commits"]:
            status = apply_commit(root, pset["repo"], sha)
            print(f"   {sha[:12]}  {status}")
            record.append(f"{pset['repo']}@{sha} {status}")
        # Post-assert the FULL probe list — a partially-present PR that failed to
        # complete, or an apply-succeeded-but-wrong-tree mistake, fails here so
        # the marker never records a patch state the tree does not have.
        still_missing = missing_probes(root, probes)
        if still_missing:
            print(f"FATAL: post-assert failed for {pset['name']}: "
                  f"{'; '.join(describe_probe(p) for p in still_missing)} "
                  f"absent under {root}", file=sys.stderr)
            return 1
        print(f"   post-assert OK — all {len(probes)} change-probes present ({pset['url']})")

    # Stale bytecode from the pre-patch install must not shadow the patched
    # sources (NeMo-RL is installed -e; Megatron rides PYTHONPATH).
    for root in set(roots.values()):
        for pycache in root.rglob("__pycache__"):
            shutil.rmtree(pycache, ignore_errors=True)

    args.marker.write_text("\n".join(record) + "\n")
    print(f"draft-PR layer complete — marker at {args.marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
