# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Cosmos-framework MFU callback subclass with a configurable peak-TFLOPS target.

WHY THIS EXISTS
---------------
The framework's ``MFUCallback.__init__`` hardcodes its hardware target as
``GB200 if is_blackwell_dc() else H100`` (H100 = 989 TFLOPS). On non-Blackwell,
non-H100 accelerators such as **p5en / H200**, this silently picks H100's peak,
so the Model FLOPs Utilization ratio is computed against the wrong denominator --
the root of an observed MFU anomaly.

``Cosmos3MFUCallback`` lets the operator override the peak-TFLOPS (and the
hardware name) via the ``peak_tflops`` kwarg or the ``COSMOS3_PEAK_TFLOPS`` env
var, making the correct H200 constant a one-line change once finalized.

NOTE: the exact H200 peak depends on precision (e.g. dense vs sparse, BF16/FP8)
and is being finalized; we deliberately DO NOT hardcode a value here. When
neither the kwarg nor the env var is set, the framework default is left intact.

Importable WITHOUT ``cosmos_framework`` (stub bases below), so it can be
unit-tested locally.
"""
from __future__ import annotations

import logging
import os

try:
    from cosmos_framework.callbacks.mfu import HardwareTarget, MFUCallback
except Exception:  # framework absent (e.g. local unit tests) -> minimal shims
    class MFUCallback:  # type: ignore
        def __init__(self, *a, **k): ...

    class HardwareTarget:  # type: ignore
        def __init__(self, name: str, peak_tflops: float):
            self.name = name
            self.peak_tflops = peak_tflops

logger = logging.getLogger(__name__)


class Cosmos3MFUCallback(MFUCallback):
    """``MFUCallback`` with an overridable peak-TFLOPS hardware target."""

    def __init__(self, *args, peak_tflops: float | None = None, hardware_name: str = "H200", **kwargs):
        super().__init__(*args, **kwargs)

        if peak_tflops is None:
            _env = os.environ.get("COSMOS3_PEAK_TFLOPS")
            if _env is not None:
                try:
                    peak_tflops = float(_env)
                except ValueError:
                    logger.warning("COSMOS3_PEAK_TFLOPS=%r is not a float; ignoring.", _env)

        if peak_tflops is not None:
            # Override the (possibly wrong) framework default with the correct
            # accelerator peak. One-line change once the H200 value is finalized.
            self.hardware_target = HardwareTarget(name=hardware_name, peak_tflops=peak_tflops)
