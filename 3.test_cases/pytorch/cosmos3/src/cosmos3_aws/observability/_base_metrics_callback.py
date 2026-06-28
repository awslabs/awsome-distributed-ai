# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Shared base for cosmos-framework metrics-export callbacks.

Holds the logic common to a metrics exporter (currently the OpenTelemetry OTLP
path): rank resolution, ``every_n`` gating, per-step time delta and loss
extraction. Concrete subclasses only implement ``_emit``.

Importable WITHOUT ``cosmos_framework`` present (the base class is imported
defensively). Observability failures must never crash the training loop.
"""
from __future__ import annotations

import logging
import os
import time

try:
    from cosmos_framework.utils.callback import Callback as _BaseCallback
except Exception:  # framework absent (e.g. local unit tests) -> minimal shim base
    class _BaseCallback:  # type: ignore
        def __init__(self, *a, **k): ...

logger = logging.getLogger(__name__)


class _BaseMetricsCallback(_BaseCallback):
    """Template callback that emits loss / step time / iteration each N steps.

    Subclasses implement :meth:`_emit` to push the values to a concrete backend.
    """

    def __init__(self, job_name: str = "cosmos3", every_n: int = 1, rank: int | None = None):
        super().__init__()
        self.job_name = job_name
        self.every_n = every_n
        if rank is None:
            rank = int(os.environ.get("RANK", "0"))
        self._is_rank0 = rank == 0
        self._last_step_time: float | None = None

    def _should_emit(self, iteration: int) -> bool:
        if not self._is_rank0:
            return False
        if self.every_n and iteration % self.every_n != 0:
            return False
        return True

    def _compute_step_time(self) -> float:
        now = time.time()
        step_time = 0.0 if self._last_step_time is None else now - self._last_step_time
        self._last_step_time = now
        return step_time

    @staticmethod
    def _loss_value(loss) -> float:
        return float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)

    def _emit(self, loss_value: float, step_time: float, iteration: int) -> None:
        raise NotImplementedError

    def on_training_step_end(self, model, data_batch: dict, output_batch: dict, loss, iteration: int = 0) -> None:
        if not self._should_emit(iteration):
            return
        step_time = self._compute_step_time()
        loss_value = self._loss_value(loss)
        try:
            self._emit(loss_value, step_time, iteration)
        except Exception as exc:  # never crash training on observability failure
            logger.warning("%s emit failed (ignored): %s", type(self).__name__, exc)

    def on_validation_end(self, model, iteration: int = 0) -> None:
        pass
