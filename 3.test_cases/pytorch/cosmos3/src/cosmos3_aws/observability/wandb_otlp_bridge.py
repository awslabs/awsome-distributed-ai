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

    _original_log = wandb.log

    def _logged_with_mirror(*args, **kwargs):
        # Mirror first (best-effort), then always call the real wandb.log.
        try:
            data = args[0] if args else kwargs.get("data")
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
        return _original_log(*args, **kwargs)

    wandb.log = _logged_with_mirror
    _installed = True
    logger.info("wandb->OTLP bridge installed (endpoint=%s, job=%s, protocol=%s).", endpoint, job_name, protocol)
    return True
