# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""GPU-free unit tests for OTLPCallback.

These tests run WITHOUT opentelemetry installed: the meter/gauge layer is
monkeypatched, which also proves the module's defensive OTLP import works.
"""
from __future__ import annotations


class _FakeLoss:
    def __init__(self, v): self._v = v
    def detach(self): return self
    def item(self): return self._v


class _FakeGauge:
    def __init__(self):
        self.values = []
    def set(self, v, *a, **k):
        self.values.append(v)


class _FakeMeter:
    def __init__(self):
        self.gauges = {}
    def create_gauge(self, name, *a, **k):
        g = _FakeGauge()
        self.gauges[name] = g
        return g


def _patch_meter(monkeypatch, mod):
    """Force the OTEL-available build path with a fake meter."""
    meter = _FakeMeter()
    monkeypatch.setattr(mod, "_OTEL_AVAILABLE", True)
    monkeypatch.setattr(mod.OTLPCallback, "_build_meter", lambda self: meter)
    return meter


def test_callback_sets_gauges_on_rank0(monkeypatch):
    import cosmos3_aws.observability.otlp_callback as mod
    meter = _patch_meter(monkeypatch, mod)

    cb = mod.OTLPCallback(endpoint="http://otel:4317", job_name="cosmos3", every_n=1, rank=0)
    cb.on_training_step_end(model=None, data_batch={}, output_batch={}, loss=_FakeLoss(0.42), iteration=5)

    assert meter.gauges["cosmos3_loss"].values == [0.42]
    assert meter.gauges["cosmos3_iteration"].values == [5]
    # step time is recorded (0.0 on the first observed step)
    assert meter.gauges["cosmos3_step_time_seconds"].values == [0.0]


def test_callback_no_set_off_rank0(monkeypatch):
    import cosmos3_aws.observability.otlp_callback as mod
    meter = _patch_meter(monkeypatch, mod)

    cb = mod.OTLPCallback(endpoint="http://otel:4317", rank=3)
    cb.on_training_step_end(model=None, data_batch={}, output_batch={}, loss=_FakeLoss(1.0), iteration=1)

    # off-rank0: no gauge ever receives a .set call
    assert all(g.values == [] for g in meter.gauges.values())


def test_callback_respects_every_n(monkeypatch):
    import cosmos3_aws.observability.otlp_callback as mod
    meter = _patch_meter(monkeypatch, mod)

    cb = mod.OTLPCallback(endpoint="http://otel:4317", every_n=5, rank=0)
    for it in range(1, 11):
        cb.on_training_step_end(model=None, data_batch={}, output_batch={}, loss=_FakeLoss(1.0), iteration=it)

    # emits at iteration 5 and 10 only
    assert meter.gauges["cosmos3_loss"].values == [1.0, 1.0]
    assert meter.gauges["cosmos3_iteration"].values == [5, 10]


def test_noop_when_otel_unavailable(monkeypatch):
    import cosmos3_aws.observability.otlp_callback as mod
    # default: _OTEL_AVAILABLE is False in this env (no opentelemetry installed)
    cb = mod.OTLPCallback(endpoint="http://otel:4317", rank=0)
    # must NOT raise even though no meter/gauges were built
    cb.on_training_step_end(model=None, data_batch={}, output_batch={}, loss=_FakeLoss(1.0), iteration=1)


def test_noop_when_endpoint_none(monkeypatch):
    import cosmos3_aws.observability.otlp_callback as mod
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setattr(mod, "_OTEL_AVAILABLE", True)  # available, but no endpoint
    cb = mod.OTLPCallback(endpoint=None, rank=0)
    # no endpoint => no meter built => emit is a no-op, must not raise
    cb.on_training_step_end(model=None, data_batch={}, output_batch={}, loss=_FakeLoss(1.0), iteration=1)


def test_set_failure_does_not_raise(monkeypatch):
    import cosmos3_aws.observability.otlp_callback as mod

    class _BoomGauge:
        def set(self, *a, **k): raise RuntimeError("export down")

    class _BoomMeter:
        def create_gauge(self, name, *a, **k): return _BoomGauge()

    monkeypatch.setattr(mod, "_OTEL_AVAILABLE", True)
    monkeypatch.setattr(mod.OTLPCallback, "_build_meter", lambda self: _BoomMeter())

    cb = mod.OTLPCallback(endpoint="http://otel:4317", rank=0)
    # a gauge .set exception must never propagate into the training loop
    cb.on_training_step_end(model=None, data_batch={}, output_batch={}, loss=_FakeLoss(1.0), iteration=1)
