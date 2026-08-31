# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""GPU-free unit tests for the wandb->OTLP bridge.

These tests run WITHOUT a real ``wandb`` or ``opentelemetry`` install: a fake
``wandb`` module is injected into ``sys.modules`` and the meter/gauge layer is
monkeypatched, which also proves the module's defensive imports work.
"""
from __future__ import annotations

import sys
import types


def test_sanitize_basic():
    import cosmos3_aws.observability.wandb_otlp_bridge as mod

    assert mod._sanitize("mfu/H100") == "cosmos3_mfu_h100"
    assert mod._sanitize("timer/iter_speed") == "cosmos3_timer_iter_speed"
    assert mod._sanitize("clip_grad_norm/video/global") == "cosmos3_clip_grad_norm_video_global"


def test_sanitize_already_prefixed_stays_sane():
    import cosmos3_aws.observability.wandb_otlp_bridge as mod

    # already-prefixed key must not gain a second prefix
    assert mod._sanitize("cosmos3_loss") == "cosmos3_loss"
    assert mod._sanitize("cosmos3/loss") == "cosmos3_loss"


def test_sanitize_non_alnum_collapses():
    import cosmos3_aws.observability.wandb_otlp_bridge as mod

    assert mod._sanitize("a.b-c d") == "cosmos3_a_b_c_d"
    # leading/trailing/duplicate separators collapse to single underscores
    assert mod._sanitize("//weird**name//") == "cosmos3_weird_name"


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


def _install_fake_wandb(monkeypatch):
    calls = []
    fake = types.ModuleType("wandb")

    def _log(*args, **kwargs):
        calls.append((args, kwargs))

    fake.log = _log
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake, calls


def _install_fake_wandb_with_run(monkeypatch):
    """Fake wandb that models the real init rebinding.

    Before init: ``wandb.log`` is a module-level pre-init shim. ``wandb.init()``
    rebinds ``wandb.log`` to the bound method ``run.log`` of a ``Run`` instance
    (exactly what wandb 0.25 does). The bridge MUST survive this rebinding.
    """
    calls = []

    class Run:
        def log(self, *args, **kwargs):
            calls.append((args, kwargs))

    fake = types.ModuleType("wandb")

    def _preinit_log(*args, **kwargs):  # pre-init shim; replaced by init
        calls.append((args, kwargs))

    fake.log = _preinit_log
    fake.run = None

    # Expose the Run class where the bridge looks for it: wandb.sdk.wandb_run.Run
    sdk = types.ModuleType("wandb.sdk")
    wandb_run = types.ModuleType("wandb.sdk.wandb_run")
    wandb_run.Run = Run
    sdk.wandb_run = wandb_run
    fake.sdk = sdk

    def _init(*args, **kwargs):
        r = Run()
        fake.run = r
        fake.log = r.log  # the rebinding that discards a wandb.log monkeypatch
        return r

    fake.init = _init

    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "wandb.sdk", sdk)
    monkeypatch.setitem(sys.modules, "wandb.sdk.wandb_run", wandb_run)
    return fake, calls, Run


def test_bridge_survives_wandb_init_rebinding(monkeypatch):
    """The real-world failure: wandb.init() rebinds wandb.log to run.log,
    discarding a wandb.log monkeypatch. Mirroring must still happen for metrics
    logged AFTER init (which is when the framework callbacks actually log)."""
    import cosmos3_aws.observability.wandb_otlp_bridge as mod

    fake_wandb, calls, _Run = _install_fake_wandb_with_run(monkeypatch)
    meter = _FakeMeter()
    _force_available(monkeypatch, mod, meter)

    assert mod.install_wandb_otlp_bridge(endpoint="http://otel:4317", job_name="cosmos3") is True

    # Framework calls wandb.init() AFTER the bridge installs.
    run = fake_wandb.init(mode="offline")

    # A framework callback logs via wandb.log (now == run.log) post-init.
    fake_wandb.log({"mfu/H200": 0.42, "timer/iter_speed": 1.7})

    # The mirror must have captured these even though wandb.log was rebound.
    assert meter.gauges["cosmos3_mfu_h200"].values == [0.42]
    assert meter.gauges["cosmos3_timer_iter_speed"].values == [1.7]


def _force_available(monkeypatch, mod, meter):
    """Make install_wandb_otlp_bridge take the success path with a fake meter."""
    monkeypatch.setattr(mod, "_OTEL_AVAILABLE", True)
    monkeypatch.setattr(mod, "_build_meter", lambda endpoint, job_name, protocol: meter)
    monkeypatch.setattr(mod, "_installed", False)


def test_wrapped_log_mirrors_only_numeric_non_bool(monkeypatch):
    import cosmos3_aws.observability.wandb_otlp_bridge as mod

    fake_wandb, calls = _install_fake_wandb(monkeypatch)
    meter = _FakeMeter()
    _force_available(monkeypatch, mod, meter)

    ok = mod.install_wandb_otlp_bridge(endpoint="http://otel:4317", job_name="cosmos3")
    assert ok is True

    fake_wandb.log({"mfu/H100": 0.5, "loss": 1.2, "table/html": "<x>", "flag": True})

    # only the two numeric non-bool values were mirrored
    assert meter.gauges["cosmos3_mfu_h100"].values == [0.5]
    assert meter.gauges["cosmos3_loss"].values == [1.2]
    assert "cosmos3_table_html" not in meter.gauges
    assert "cosmos3_flag" not in meter.gauges  # bool is excluded

    # the original wandb.log was still called once with the same payload
    assert len(calls) == 1
    assert calls[0][0][0] == {"mfu/H100": 0.5, "loss": 1.2, "table/html": "<x>", "flag": True}


def test_noop_when_unavailable(monkeypatch):
    import cosmos3_aws.observability.wandb_otlp_bridge as mod

    # no fake wandb in sys.modules, OTEL unavailable -> returns False, no raise
    monkeypatch.setitem(sys.modules, "wandb", None) if "wandb" in sys.modules else None
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    monkeypatch.setattr(mod, "_OTEL_AVAILABLE", False)
    monkeypatch.setattr(mod, "_installed", False)

    assert mod.install_wandb_otlp_bridge(endpoint="http://otel:4317") is False


def test_idempotent_install(monkeypatch):
    import cosmos3_aws.observability.wandb_otlp_bridge as mod

    fake_wandb, _ = _install_fake_wandb(monkeypatch)
    meter = _FakeMeter()
    _force_available(monkeypatch, mod, meter)

    assert mod.install_wandb_otlp_bridge(endpoint="http://otel:4317") is True
    first_log = fake_wandb.log
    # second call must NOT re-wrap (log identity unchanged) and still report True
    assert mod.install_wandb_otlp_bridge(endpoint="http://otel:4317") is True
    assert fake_wandb.log is first_log


def test_mirror_failure_does_not_break_log(monkeypatch):
    import cosmos3_aws.observability.wandb_otlp_bridge as mod

    fake_wandb, calls = _install_fake_wandb(monkeypatch)

    class _BoomGauge:
        def set(self, *a, **k):
            raise RuntimeError("export down")

    class _BoomMeter:
        def create_gauge(self, name, *a, **k):
            return _BoomGauge()

    _force_available(monkeypatch, mod, _BoomMeter())
    assert mod.install_wandb_otlp_bridge(endpoint="http://otel:4317") is True

    # mirror failure must never propagate; original log still runs
    fake_wandb.log({"loss": 1.0})
    assert len(calls) == 1
