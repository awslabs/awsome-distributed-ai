# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""GPU-free unit tests for PrometheusCallback (no live gateway, no framework)."""
from __future__ import annotations


class _FakeLoss:
    def __init__(self, v): self._v = v
    def detach(self): return self
    def item(self): return self._v


def test_callback_pushes_step_metrics_on_rank0(monkeypatch):
    import cosmos3_aws.observability.prometheus_callback as mod

    pushed = {}
    def fake_push(gateway, job, registry, **kw):
        pushed["gateway"] = gateway
        pushed["job"] = job
        pushed["registry"] = registry
    monkeypatch.setattr(mod, "push_to_gateway", fake_push)

    cb = mod.PrometheusCallback(pushgateway_url="http://pgw:9091", job_name="cosmos3", every_n=1, rank=0)
    cb.on_training_step_end(model=None, data_batch={}, output_batch={}, loss=_FakeLoss(0.42), iteration=5)

    assert pushed["job"] == "cosmos3"
    assert "pgw:9091" in pushed["gateway"]


def test_callback_no_push_off_rank0(monkeypatch):
    import cosmos3_aws.observability.prometheus_callback as mod

    pushed = {"called": False}
    def fake_push(*a, **k):
        pushed["called"] = True
    monkeypatch.setattr(mod, "push_to_gateway", fake_push)

    cb = mod.PrometheusCallback(pushgateway_url="http://pgw:9091", rank=3)
    cb.on_training_step_end(model=None, data_batch={}, output_batch={}, loss=_FakeLoss(1.0), iteration=1)
    assert pushed["called"] is False


def test_callback_respects_every_n(monkeypatch):
    import cosmos3_aws.observability.prometheus_callback as mod
    pushed = {"n": 0}
    monkeypatch.setattr(mod, "push_to_gateway", lambda *a, **k: pushed.__setitem__("n", pushed["n"] + 1))
    cb = mod.PrometheusCallback(pushgateway_url="http://pgw:9091", every_n=5, rank=0)
    for it in range(1, 11):
        cb.on_training_step_end(model=None, data_batch={}, output_batch={}, loss=_FakeLoss(1.0), iteration=it)
    # pushes at iteration 5 and 10 only
    assert pushed["n"] == 2


def test_push_failure_does_not_raise(monkeypatch):
    import cosmos3_aws.observability.prometheus_callback as mod
    def boom(*a, **k): raise OSError("gateway down")
    monkeypatch.setattr(mod, "push_to_gateway", boom)
    cb = mod.PrometheusCallback(pushgateway_url="http://pgw:9091", rank=0)
    # must NOT raise — observability failures cannot crash training
    cb.on_training_step_end(model=None, data_batch={}, output_batch={}, loss=_FakeLoss(1.0), iteration=1)
