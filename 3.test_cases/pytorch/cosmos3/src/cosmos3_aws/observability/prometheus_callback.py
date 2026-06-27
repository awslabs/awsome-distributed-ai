# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Reusable cosmos-framework callback that bridges trainer metrics (loss, step
time, iteration) into a Prometheus Pushgateway, so they can be scraped into
Amazon Managed Prometheus and unified with GPU/DCGM metrics in Grafana.

The module is intentionally importable WITHOUT ``cosmos_framework`` present (the
base class is imported defensively in :mod:`._base_metrics_callback`) and WITHOUT
a live gateway (pushes are wrapped in try/except), so it can be unit-tested
locally. Observability failures must never crash the training loop.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from cosmos3_aws.observability._base_metrics_callback import _BaseMetricsCallback


class PrometheusCallback(_BaseMetricsCallback):
    """Push training loss / step time / iteration to a Prometheus Pushgateway."""

    def __init__(self, pushgateway_url: str, job_name: str = "cosmos3", every_n: int = 1, rank: int | None = None):
        super().__init__(job_name=job_name, every_n=every_n, rank=rank)
        self.pushgateway_url = pushgateway_url

        self._registry = CollectorRegistry()
        self._loss = Gauge("cosmos3_loss", "Training loss", registry=self._registry)
        self._step_time = Gauge("cosmos3_step_time_seconds", "Seconds per training step", registry=self._registry)
        self._iteration = Gauge("cosmos3_iteration", "Current training iteration", registry=self._registry)

    def _emit(self, loss_value: float, step_time: float, iteration: int) -> None:
        self._loss.set(loss_value)
        self._step_time.set(step_time)
        self._iteration.set(iteration)
        push_to_gateway(self.pushgateway_url, job=self.job_name, registry=self._registry)
