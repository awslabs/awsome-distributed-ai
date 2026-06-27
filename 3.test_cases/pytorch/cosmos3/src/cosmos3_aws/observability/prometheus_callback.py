# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Reusable cosmos-framework callback that bridges trainer metrics (loss, step
time, iteration) into a Prometheus Pushgateway, so they can be scraped into
Amazon Managed Prometheus and unified with GPU/DCGM metrics in Grafana.

The module is intentionally importable WITHOUT ``cosmos_framework`` present (the
base class is imported defensively) and WITHOUT a live gateway (pushes are wrapped
in try/except), so it can be unit-tested locally. Observability failures must never
crash the training loop.
"""
from __future__ import annotations

import logging
import os
import time

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

try:
    from cosmos_framework.utils.callback import Callback as _BaseCallback
except Exception:  # framework absent (e.g. local unit tests) -> minimal shim base
    class _BaseCallback:  # type: ignore
        def __init__(self, *a, **k): ...

logger = logging.getLogger(__name__)


class PrometheusCallback(_BaseCallback):
    """Push training loss / step time / iteration to a Prometheus Pushgateway."""

    def __init__(self, pushgateway_url: str, job_name: str = "cosmos3", every_n: int = 1, rank: int | None = None):
        super().__init__()
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        self.every_n = every_n
        if rank is None:
            rank = int(os.environ.get("RANK", "0"))
        self._is_rank0 = rank == 0
        self._last_step_time: float | None = None

        self._registry = CollectorRegistry()
        self._loss = Gauge("cosmos3_loss", "Training loss", registry=self._registry)
        self._step_time = Gauge("cosmos3_step_time_seconds", "Seconds per training step", registry=self._registry)
        self._iteration = Gauge("cosmos3_iteration", "Current training iteration", registry=self._registry)

    def on_training_step_end(self, model, data_batch: dict, output_batch: dict, loss, iteration: int = 0) -> None:
        if not self._is_rank0:
            return
        if self.every_n and iteration % self.every_n != 0:
            return

        now = time.time()
        step_time = 0.0 if self._last_step_time is None else now - self._last_step_time
        self._last_step_time = now

        loss_value = float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)
        self._loss.set(loss_value)
        self._step_time.set(step_time)
        self._iteration.set(iteration)

        try:
            push_to_gateway(self.pushgateway_url, job=self.job_name, registry=self._registry)
        except Exception as exc:  # never crash training on observability failure
            logger.warning("PrometheusCallback push failed (ignored): %s", exc)

    def on_validation_end(self, model, iteration: int = 0) -> None:
        pass
