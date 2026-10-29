# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""GPU-free unit tests for Cosmos3MFUCallback.

These tests run WITHOUT ``cosmos_framework`` installed: the module falls back to
stub ``MFUCallback`` / ``HardwareTarget`` bases, which also proves the defensive
import path works.
"""
from __future__ import annotations


def test_peak_tflops_kwarg_overrides_hardware_target():
    import cosmos3_aws.observability.cosmos3_mfu_callback as mod

    cb = mod.Cosmos3MFUCallback(peak_tflops=1979.0, hardware_name="H200")
    assert cb.hardware_target.peak_tflops == 1979.0
    assert cb.hardware_target.name == "H200"


def test_peak_tflops_read_from_env(monkeypatch):
    import cosmos3_aws.observability.cosmos3_mfu_callback as mod

    monkeypatch.setenv("COSMOS3_PEAK_TFLOPS", "1500.5")
    cb = mod.Cosmos3MFUCallback(hardware_name="H200")
    assert cb.hardware_target.peak_tflops == 1500.5
    assert cb.hardware_target.name == "H200"


def test_none_leaves_base_default_untouched(monkeypatch):
    import cosmos3_aws.observability.cosmos3_mfu_callback as mod

    monkeypatch.delenv("COSMOS3_PEAK_TFLOPS", raising=False)
    cb = mod.Cosmos3MFUCallback()
    base = mod.MFUCallback()
    # With neither kwarg nor env, we must NOT override: hardware_target is whatever
    # the base set — None under the local stub, the framework default (e.g. H100)
    # in-image. Comparing to a bare base instance keeps this valid in both.
    assert getattr(cb, "hardware_target", None) == getattr(base, "hardware_target", None)
