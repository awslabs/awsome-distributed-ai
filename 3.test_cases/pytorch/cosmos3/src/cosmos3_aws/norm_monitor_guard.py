# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Sample-side guard for the framework's NormMonitor callback on empty LoRA shards.

WHY THIS EXISTS
---------------
The framework's ``NormMonitor`` callback (telemetry: per-parameter L2/max gradient
stats) computes ``data.abs().max()`` on each rank's *local* FSDP shard in
``_compute_l2_stats``. For a **LoRA** run (only the small ``lora_`` parameter set is
trainable), once the rank count is high enough that the LoRA parameters shard to an
**empty** local tensor on some ranks, ``torch.max()`` on a 0-element tensor raises::

    RuntimeError: max(): Expected reduction dim to be specified for input.numel() == 0

This fires at global_step 0 (``0 % every_n == 0`` for any ``every_n``), so it cannot
be throttled, and it is composed in from the ``basic`` callbacks group so it is awkward
to remove via a Hydra ``~`` override. It is the true root cause of the long-observed
"Super vision-LoRA crashes at >= 4 nodes" symptom (a full-finetune model like Nano never
hits it because no rank gets an empty parameter shard).

This module monkeypatches ``NormMonitor._compute_l2_stats`` to return zero-valued
scalars when the local shard is empty, instead of calling reductions on a 0-element
tensor. It is a sample-side, no-framework-edit guard: importing this module installs
the patch as an import side-effect. Telemetry for empty shards is simply reported as
zero (correct: an empty shard contributes nothing to the global all-reduced norm/max).
"""
from __future__ import annotations

import torch

from cosmos_framework.callbacks.norm_monitor import NormMonitor

try:  # DTensor handling mirrors the framework
    from torch.distributed.tensor import DTensor
except Exception:  # pragma: no cover - older torch layout
    from torch.distributed._tensor import DTensor  # type: ignore


def _safe_compute_l2_stats(self, tensor: torch.Tensor, detach: bool = True) -> dict:
    data = tensor.detach() if detach else tensor
    if isinstance(data, DTensor):
        data = data.to_local()
    if data.numel() == 0:
        # Empty local shard (e.g. LoRA params sharded across more ranks than there
        # are LoRA tensors). Contribute zero to the all-reduced norm/max.
        z = torch.zeros((), device=data.device, dtype=torch.float32)
        return {"sq_sum": z, "max": z}
    return {
        "sq_sum": (data.float() ** 2).sum(),
        "max": data.abs().max(),
    }


NormMonitor._compute_l2_stats = _safe_compute_l2_stats
