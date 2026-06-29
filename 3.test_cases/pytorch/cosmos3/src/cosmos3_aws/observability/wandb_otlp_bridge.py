# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Runtime bridge that mirrors every numeric scalar logged via ``wandb.log`` to
OpenTelemetry OTLP gauges.

WHY THIS EXISTS
---------------
The cosmos-framework callbacks (MFU, iter_speed, sequence-packing, grad_clip,
data_stats, ...) report their metrics through ``wandb.log({...})``. The OTLP
callback only exports the three trainer scalars it computes itself (loss, step
time, iteration). To land the *framework's* rich metric set in Amazon Managed
Prometheus -- without editing the framework -- this module monkeypatches
``wandb.log`` so each numeric scalar it receives is ALSO ``gauge.set()`` onto an
OTLP MeterProvider pointed at the HyperPod observability addon's OTLP receiver.

SURVIVING ``wandb.init()`` (load-bearing)
-----------------------------------------
The bridge installs BEFORE the framework calls ``wandb.init()``. But
``wandb.init(force=True, ...)`` rebinds the module-level ``wandb.log`` to the
freshly-created run's bound method ``run.log`` -- which would DISCARD a patch
applied only to ``wandb.log``. Since the framework callbacks log AFTER init,
none of their metrics would be mirrored (verified live on p5en: only the
OTLPCallback's own gauges reached the sink). The fix wraps
``wandb.sdk.wandb_run.Run.log`` at the CLASS level too: because post-init
``wandb.log`` IS ``run.log``, the class-level wrap survives re-init. The
module-level patch is kept for any pre-init logging.

This is a documented, sample-side runtime bridge, in the same spirit as
``norm_monitor_guard.py`` (a framework monkeypatch installed via
``sitecustomize.py``). It is importable WITHOUT ``wandb`` or ``opentelemetry``
present and degrades to a no-op when either is missing. Mirror failures are
swallowed: the bridge must never break training or the real ``wandb.log``.
"""
from __future__ import annotations

import logging
import os
import re

try:
    from opentelemetry import metrics as _otel_metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    # grpc + http exporters are imported lazily in _build_meter based on protocol
    _OTEL_AVAILABLE = True
except Exception:  # opentelemetry absent (e.g. local unit tests) -> no-op bridge
    _OTEL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Module-level state guarding idempotent install + holding lazily-created gauges.
_installed = False
_meter = None
_gauges: dict = {}

_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def _sanitize(key: str) -> str:
    """Normalize a wandb metric key into a Prometheus-safe ``cosmos3_*`` name.

    ``mfu/H100`` -> ``cosmos3_mfu_h100``; ``/`` and any non-alphanumeric run
    collapse to a single ``_``; the result is lowercased and prefixed with
    ``cosmos3_`` unless it already starts with it.
    """
    s = _NON_ALNUM.sub("_", str(key).lower()).strip("_")
    if not s.startswith("cosmos3_"):
        s = "cosmos3_" + s
    return s


def _build_meter(endpoint: str, job_name: str, protocol: str):
    """Build a MeterProvider with an OTLP exporter and return a Meter.

    Kept as a seam so unit tests can monkeypatch the meter/gauge layer without a
    real opentelemetry install.
    """
    if protocol == "http":
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    else:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

    attributes = {"service.name": job_name}
    cluster_name = os.environ.get("CLUSTER_NAME")
    if cluster_name:
        attributes["cluster_name"] = cluster_name
    resource = Resource.create(attributes)

    exporter = OTLPMetricExporter(endpoint=endpoint)
    reader = PeriodicExportingMetricReader(exporter)
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    _otel_metrics.set_meter_provider(provider)
    return provider.get_meter(job_name)


def install_wandb_otlp_bridge(endpoint: str, job_name: str = "cosmos3", protocol: str = "grpc") -> bool:
    """Wrap ``wandb.log`` so numeric scalars are mirrored to OTLP gauges.

    Returns ``True`` if the bridge was installed (or was already installed),
    ``False`` if ``wandb`` or the OTEL SDK is unavailable / the build failed. Never
    raises -- observability must not crash training.
    """
    global _installed, _meter

    if _installed:
        return True

    if not _OTEL_AVAILABLE:
        logger.warning("wandb->OTLP bridge: opentelemetry SDK unavailable; bridge disabled (no-op).")
        return False

    try:
        import wandb  # defensive: wandb may be absent in some environments
    except Exception:
        logger.warning("wandb->OTLP bridge: wandb unavailable; bridge disabled (no-op).")
        return False

    try:
        _meter = _build_meter(endpoint, job_name, protocol)
    except Exception as exc:  # build failure -> stay a no-op, never crash training
        logger.warning("wandb->OTLP bridge: meter build failed (bridge disabled): %s", exc)
        return False

    def _mirror(data) -> None:
        # Mirror numeric scalars to OTLP gauges (best-effort, never raises).
        try:
            if isinstance(data, dict):
                for k, v in data.items():
                    # numeric scalars only; bool is a subclass of int -> exclude it
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        continue
                    name = _sanitize(k)
                    gauge = _gauges.get(name)
                    if gauge is None:
                        gauge = _meter.create_gauge(name)
                        _gauges[name] = gauge
                    gauge.set(v)
        except Exception as exc:  # a mirror failure must never break real logging
            logger.debug("wandb->OTLP bridge: mirror failed (ignored): %s", exc)

    # Wrap the module-level ``wandb.log`` for any pre-init logging.
    _original_log = wandb.log

    def _logged_with_mirror(*args, **kwargs):
        _mirror(args[0] if args else kwargs.get("data"))
        return _original_log(*args, **kwargs)

    wandb.log = _logged_with_mirror

    # CRITICAL: ``wandb.init()`` rebinds the module-level ``wandb.log`` to the
    # bound method ``run.log`` of the freshly-created Run, which DISCARDS the
    # module-level monkeypatch above. The framework's callbacks (MFU, iter_speed,
    # sequence-packing, grad_clip, ...) all log AFTER ``wandb.init()``, so without
    # also wrapping at the class level their metrics would never be mirrored.
    # Wrapping ``Run.log`` on the class survives re-init because the rebound
    # ``wandb.log`` IS ``run.log`` -> our wrapped class method. Idempotent via a
    # sentinel attribute so repeated installs don't stack wrappers.
    try:
        from wandb.sdk.wandb_run import Run as _WandbRun

        if not getattr(_WandbRun.log, "_cosmos3_otlp_wrapped", False):
            _original_run_log = _WandbRun.log

            def _run_log_with_mirror(self, *args, **kwargs):
                _mirror(args[0] if args else kwargs.get("data"))
                return _original_run_log(self, *args, **kwargs)

            _run_log_with_mirror._cosmos3_otlp_wrapped = True  # type: ignore[attr-defined]
            _WandbRun.log = _run_log_with_mirror  # type: ignore[assignment]
    except Exception as exc:  # Run class unavailable / wandb internals shifted -> module-level patch stands
        logger.warning("wandb->OTLP bridge: Run.log wrap skipped (%s); module-level wandb.log patch still active.", exc)

    _installed = True
    logger.info("wandb->OTLP bridge installed (endpoint=%s, job=%s, protocol=%s).", endpoint, job_name, protocol)
    return True
