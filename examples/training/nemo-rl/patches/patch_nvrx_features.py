#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Runtime patches to enable NVRx features in NeMo RL container.

Applied by the K8s manifest entrypoint BEFORE ft_launcher starts.
After patching, all __pycache__ directories under /opt/nemo-rl are deleted
so Python recompiles from the patched .py files.

Features enabled:
  1. Async checkpoint: adds is_async to CheckpointingConfig TypedDict
  2. Straggler detection: wraps forward_backward with NVRx Detector in
     DTensorPolicyWorkerV2

Design rationale (per PR #1010 review by @paragao):
  We chose runtime source-file patching over (a) baking patches into the
  Dockerfile or (b) forking NeMo RL because:
    * The pinned NeMo RL commit (NRL_GIT_REF in the Dockerfile) is the only
      thing guaranteed to be paired with these patches; moving them into the
      image would make image rebuilds the only way to experiment with a new
      NRL commit during the workshop.
    * Two of the three features (is_async TypedDict field, straggler section
      wrapping) are already being upstreamed; inlining them in a fork would
      create long-lived fork drift.
    * The patches are idempotent and fail loudly (each function returns False
      and prints [NVRx patch] FAIL: ... when a string anchor is missing) so
      a future NeMo RL revision that drifts from the anchors cannot silently
      ship a broken container.
  This file is intentionally scoped to the workshop — do not import runtime
  patching as a pattern for production clusters.
"""

import os
import re
import shutil
import sys

NEMO_RL_ROOT = "/opt/nemo-rl"


def patch_checkpoint_config():
    """Add is_async: NotRequired[bool] to CheckpointingConfig TypedDict.

    Target: nemo_rl/utils/checkpoint.py
    The is_async parameter is already consumed by AutomodelCheckpointManager
    but the TypedDict doesn't declare it, so Hydra rejects the config override.
    """
    target = os.path.join(NEMO_RL_ROOT, "nemo_rl/utils/checkpoint.py")
    if not os.path.exists(target):
        print(f"[NVRx patch] SKIP checkpoint config: {target} not found")
        return False

    with open(target) as f:
        content = f.read()

    if "is_async" in content:
        print("[NVRx patch] checkpoint config: is_async already present, skipping")
        return True

    # Insert after the peft_config line
    marker = '    peft_config: NotRequired[Any]  # Default: None'
    if marker not in content:
        # Try a looser match
        marker = '    peft_config: NotRequired[Any]'
    if marker not in content:
        print(f"[NVRx patch] WARN: could not find peft_config line in {target}")
        # Try inserting before the class CheckpointManager line instead
        marker2 = '\nclass CheckpointManager:'
        if marker2 not in content:
            print(f"[NVRx patch] FAIL: no insertion point found in {target}")
            return False
        insert = '    is_async: NotRequired[bool]  # NVRx: enable async checkpoint writes\n\n'
        content = content.replace(marker2, insert + marker2)
    else:
        insert = '\n    is_async: NotRequired[bool]  # NVRx: enable async checkpoint writes'
        content = content.replace(marker, marker + insert)

    with open(target, "w") as f:
        f.write(content)
    print(f"[NVRx patch] OK: added is_async to CheckpointingConfig in {target}")
    return True


def patch_straggler_detection():
    """Add NVRx straggler detection to DTensorPolicyWorkerV2.

    Target: nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py

    Changes:
      - Import nvidia_resiliency_ext straggler Detector (with try/except)
      - Initialize Detector in __init__ after distributed setup
      - Wrap automodel_forward_backward call in train() with detection_section
      - Log straggler report every 5 steps
    """
    target = os.path.join(
        NEMO_RL_ROOT,
        "nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py",
    )
    if not os.path.exists(target):
        print(f"[NVRx patch] SKIP straggler detection: {target} not found")
        return False

    with open(target) as f:
        content = f.read()

    if "straggler" in content.lower() and "nvidia_resiliency_ext" in content:
        print("[NVRx patch] straggler detection: already patched, skipping")
        return True

    # --- 1. Add import block after existing imports ---
    # NOTE: `nullcontext` is used below when the straggler detector is unavailable
    # or disabled — it MUST be imported here, otherwise the else-branch raises
    # NameError the first time forward_backward runs. (PR #1010 review, @paragao)
    import_block = '''
# NVRx straggler detection (runtime patch)
from contextlib import nullcontext
try:
    from nvidia_resiliency_ext.straggler import Detector as _NVRxStragglerDetector
    _NVRX_STRAGGLER_AVAILABLE = True
except ImportError:
    _NVRX_STRAGGLER_AVAILABLE = False
'''
    # Insert after the last nemo_rl import block
    anchor = "from nemo_rl.utils.packed_tensor import packed_broadcast_producer"
    if anchor not in content:
        # Fallback: insert after the nsys import
        anchor = "from nemo_rl.utils.nsys import wrap_with_nvtx_name"
    if anchor not in content:
        print(f"[NVRx patch] FAIL: no import anchor found in {target}")
        return False
    content = content.replace(anchor, anchor + "\n" + import_block)

    # --- 2. Initialize Detector in __init__ after distributed setup ---
    # Insert after: self.cp_size = distributed_manager.cp_size
    init_anchor = "        self.cp_size = distributed_manager.cp_size"
    if init_anchor not in content:
        # Try alternative
        init_anchor = "self.cp_size = distributed_manager.cp_size"
    if init_anchor not in content:
        print(f"[NVRx patch] FAIL: no __init__ anchor found in {target}")
        return False

    init_block = '''

        # NVRx straggler detection (runtime patch)
        # Detector is a class-level singleton — all methods are @classmethod
        self._nvrx_straggler_enabled = False
        self._nvrx_train_step_count = 0
        if _NVRX_STRAGGLER_AVAILABLE:
            try:
                _NVRxStragglerDetector.initialize(
                    scores_to_compute="all",
                    gather_on_rank0=True,
                    report_time_interval=120,
                )
                self._nvrx_straggler_enabled = True
                print(f"[NVRx] Straggler detector initialized on rank {torch.distributed.get_rank()}")
            except Exception as e:
                print(f"[NVRx] Straggler detector init failed: {e}")'''

    # Use the full line with leading spaces for replacement
    content = content.replace(
        "        self.cp_size = distributed_manager.cp_size",
        "        self.cp_size = distributed_manager.cp_size" + init_block,
        1,  # only first occurrence
    )

    # --- 3. Wrap automodel_forward_backward in train() ---
    # Find the automodel_forward_backward call and wrap it
    fwd_bwd_call = "                mb_results = automodel_forward_backward("
    if fwd_bwd_call not in content:
        print(f"[NVRx patch] FAIL: automodel_forward_backward call not found in {target}")
        return False

    # We need to wrap the whole call. Find the closing paren.
    # Strategy: replace the assignment line with a context-managed version
    wrapped = '''                # NVRx straggler detection wrapping (runtime patch)
                _nvrx_ctx = (
                    _NVRxStragglerDetector.detection_section("forward_backward")
                    if self._nvrx_straggler_enabled
                    else nullcontext()
                )
                with _nvrx_ctx:
                    mb_results = automodel_forward_backward('''
    content = content.replace(fwd_bwd_call, wrapped, 1)

    # Indent the body of automodel_forward_backward call by 4 more spaces
    # Find the lines between "mb_results = automodel_forward_backward(" and the closing ")"
    # We need to indent the keyword args. They follow the pattern "                    key=value,"
    # The closing paren is on a line by itself: "                )"

    # Find where the call arguments end. Look for the line with just "                )"
    # after the automodel_forward_backward call.
    # The args are already indented at 20 spaces. We need them at 24.
    # Actually, since we wrapped with "with _nvrx_ctx:\n    mb_results = ...",
    # the args need 4 extra spaces of indent.
    lines = content.split('\n')
    in_fwd_bwd = False
    new_lines = []
    paren_depth = 0
    for i, line in enumerate(lines):
        if 'mb_results = automodel_forward_backward(' in line and '_nvrx_ctx' not in line:
            in_fwd_bwd = True
            paren_depth = line.count('(') - line.count(')')
            new_lines.append(line)
            continue
        if in_fwd_bwd:
            paren_depth += line.count('(') - line.count(')')
            new_lines.append('    ' + line)
            if paren_depth <= 0:
                in_fwd_bwd = False
            continue
        new_lines.append(line)
    content = '\n'.join(new_lines)

    # --- 4. Add straggler report logging after metrics aggregation ---
    report_anchor = "            return metrics"
    report_block = '''
            # NVRx straggler report (runtime patch)
            self._nvrx_train_step_count += 1
            if (
                self._nvrx_straggler_enabled
                and self._nvrx_train_step_count % 5 == 0
            ):
                try:
                    report = _NVRxStragglerDetector.generate_report()
                    if torch.distributed.get_rank() == 0:
                        print(f"[NVRx] Straggler report (step {self._nvrx_train_step_count}):")
                        print(report)
                except Exception as e:
                    print(f"[NVRx] Straggler report failed: {e}")

'''
    # Find the return metrics that's inside train() method (first occurrence after train def)
    # Make sure we only patch the one inside train()
    train_def_pos = content.find("def train(")
    if train_def_pos == -1:
        print(f"[NVRx patch] FAIL: train() method not found")
        return False
    return_pos = content.find(report_anchor, train_def_pos)
    if return_pos == -1:
        print(f"[NVRx patch] FAIL: 'return metrics' not found in train()")
        return False
    content = content[:return_pos] + report_block + content[return_pos:]

    with open(target, "w") as f:
        f.write(content)
    print(f"[NVRx patch] OK: added straggler detection to {target}")
    return True


def cleanup_pycache():
    """Delete all __pycache__ directories under NEMO_RL_ROOT.

    Python caches compiled .pyc files. After patching .py source,
    stale .pyc would be loaded instead of the patched code.
    """
    count = 0
    for root, dirs, _files in os.walk(NEMO_RL_ROOT):
        for d in dirs:
            if d == "__pycache__":
                path = os.path.join(root, d)
                shutil.rmtree(path, ignore_errors=True)
                count += 1
    print(f"[NVRx patch] Deleted {count} __pycache__ directories under {NEMO_RL_ROOT}")


def main():
    print("=" * 60)
    print("[NVRx] Applying runtime patches to NeMo RL")
    print("=" * 60)

    results = {}
    results["checkpoint_config"] = patch_checkpoint_config()
    results["straggler_detection"] = patch_straggler_detection()

    cleanup_pycache()

    print()
    print("[NVRx] Patch summary:")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")

    all_ok = all(results.values())
    if all_ok:
        print("[NVRx] All patches applied successfully")
    else:
        print("[NVRx] WARNING: Some patches failed (training may still work)")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
