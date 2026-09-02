# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Cosmos-framework callback that exports trainer metrics (loss, step time,
iteration) via OpenTelemetry OTLP.

Metrics reach the HyperPod observability addon's OTLP receiver directly, with no
Pushgateway and no collector-config edit, where they are remote-written into
Amazon Managed Prometheus and unified with GPU/DCGM metrics in Grafana. A
``PeriodicExportingMetricReader`` handles export in the background, so
:meth:`_emit` only updates gauge values.

The module is importable WITHOUT ``opentelemetry`` installed (the SDK is imported
defensively) and degrades to a no-op when the SDK or an OTLP endpoint is missing.
Observability failures must never crash the training loop.
"""
from __future__ import annotations

import logging
import os

from cosmos3_aws.observability._base_metrics_callback import _BaseMetricsCallback

try:
    from opentelemetry import metrics as _otel_metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    # grpc + http exporters are imported lazily in _build_meter based on protocol
    _OTEL_AVAILABLE = True
except Exception:  # opentelemetry absent (e.g. local unit tests) -> no-op exporter
    _OTEL_AVAILABLE = False

logger = logging.getLogger(__name__)


class OTLPCallback(_BaseMetricsCallback):
    """Export training loss / step time / iteration via OpenTelemetry OTLP."""

    def __init__(
        self,
        endpoint: str | None = None,
        job_name: str = "cosmos3",
        every_n: int = 1,
        rank: int | None = None,
        protocol: str | None = "grpc",
    ):
        super().__init__(job_name=job_name, every_n=every_n, rank=rank)
        self.endpoint = endpoint if endpoint is not None else os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        self.protocol = protocol if protocol is not None else os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")

        self._loss_gauge = None
        self._step_time_gauge = None
        self._iteration_gauge = None

        if _OTEL_AVAILABLE and self.endpoint and self._is_rank0:
            try:
                meter = self._build_meter()
                self._loss_gauge = meter.create_gauge("cosmos3_loss")
                self._step_time_gauge = meter.create_gauge("cosmos3_step_time_seconds")
                self._iteration_gauge = meter.create_gauge("cosmos3_iteration")
            except Exception as exc:  # build failure -> stay a no-op, never crash training
                logger.warning("OTLPCallback meter build failed (metrics disabled): %s", exc)
                self._loss_gauge = None
                self._step_time_gauge = None
                self._iteration_gauge = None

    def _build_meter(self):
        """Build a MeterProvider with an OTLP exporter and return a Meter.

        Kept as a seam so unit tests can monkeypatch the meter/gauge layer
        without a real opentelemetry install.
        """
        if self.protocol == "http":
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        else:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

        attributes = {"service.name": self.job_name}
        cluster_name = os.environ.get("CLUSTER_NAME")
        if cluster_name:
            attributes["cluster_name"] = cluster_name
        resource = Resource.create(attributes)

        exporter = OTLPMetricExporter(endpoint=self.endpoint)
        reader = PeriodicExportingMetricReader(exporter)
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        _otel_metrics.set_meter_provider(provider)
        return provider.get_meter(self.job_name)

    def _emit(self, loss_value: float, step_time: float, iteration: int) -> None:
        if self._loss_gauge is None:
            return  # no endpoint / OTEL unavailable / build failed -> no-op
        self._loss_gauge.set(loss_value)
        self._step_time_gauge.set(step_time)
        self._iteration_gauge.set(iteration)
