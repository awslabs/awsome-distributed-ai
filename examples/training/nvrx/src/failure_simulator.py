"""
Configurable Fault Injector for NVRx Resiliency Testing

Supports three fault types that exercise different recovery paths:

  exception  -- raise RuntimeError. Caught by NVRx in-process Wrapper
               (sub-second recovery) or causes worker crash for ft_launcher
               in-job restart.

  sigkill    -- os.kill(os.getpid(), SIGKILL). Instantly kills the worker
               process. In-process Wrapper cannot catch this; only
               ft_launcher in-job restart (or K8s container restart in
               baseline mode) can recover.

  hang       -- time.sleep(999999). Simulates a stuck process (e.g. NCCL
               deadlock, GPU hang). Detected by in-process Wrapper's soft
               timeout or ft_launcher's heartbeat timeout.

Fault injection modes:

  Stochastic (--fault_probability):
    Each rank independently rolls per step. Total faults vary between runs.

  Deterministic (--fault_count + --fault_seed):
    Pre-generates exactly N faults at random steps, each targeting a random
    non-zero rank. Both experiments using the same seed get identical fault
    patterns, making the comparison fair. Ignores --fault_probability.

Usage (stochastic):
    injector = FaultInjector(
        fault_types=["exception", "sigkill"],
        weights=[0.7, 0.3],
        probability=0.002,
        after_step=50,
    )

Usage (deterministic):
    injector = FaultInjector(
        fault_types=["exception"],
        fault_count=5,
        fault_seed=42,
        max_steps=1000,
        world_size=16,
        after_step=10,
    )

    # Inside training loop:
    injector.maybe_inject(step, rank)
"""

import os
import time
import random
import signal
import logging
from typing import Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

VALID_FAULT_TYPES = {"exception", "sigkill", "hang"}


class FaultInjector:
    """
    Fault injector for distributed training resilience testing.

    Supports two modes:

    **Stochastic mode** (default): Each rank independently rolls
    ``random() < probability`` at every step. Total fault count varies
    between runs.

    **Deterministic mode** (``fault_count`` set): Pre-generates exactly N
    faults at random steps, each targeting a specific non-zero rank. Using
    the same ``fault_seed`` across experiments guarantees identical fault
    patterns, making comparisons fair.

    Faults are only injected on non-zero ranks (rank 0 hosts the rendezvous
    store and killing it would take down the entire job rather than testing
    per-worker recovery).

    Args:
        fault_types: List of fault types to inject. Valid values:
            "exception", "sigkill", "hang".
        weights: Optional list of weights (same length as fault_types).
            Controls relative probability of each fault type.  Defaults
            to uniform distribution.
        probability: Per-step probability of injecting a fault (stochastic mode).
        after_step: Only inject faults after this step number (allow
            initial training to stabilise).
        on_fault: Optional callback(step, rank, fault_type) called before injection.
        fault_count: If set, use deterministic mode with exactly this many faults.
            Overrides ``probability``.
        fault_seed: Random seed for deterministic fault schedule generation.
            Only used when ``fault_count`` is set.
        max_steps: Total training steps (required for deterministic mode to
            generate fault steps within range).
        world_size: Total number of ranks (required for deterministic mode to
            assign target ranks).
    """

    def __init__(
        self,
        fault_types: List[str],
        weights: Optional[List[float]] = None,
        probability: float = 0.002,
        after_step: int = 50,
        on_fault: Optional[Callable] = None,
        fault_count: Optional[int] = None,
        fault_seed: int = 42,
        max_steps: int = 1000,
        world_size: int = 2,
        shared_store=None,
        pre_injected_steps: Optional[List[int]] = None,
    ):
        # Validate fault types
        for ft in fault_types:
            if ft not in VALID_FAULT_TYPES:
                raise ValueError(
                    f"Invalid fault type '{ft}'. "
                    f"Valid types: {sorted(VALID_FAULT_TYPES)}"
                )
        if len(fault_types) == 0:
            raise ValueError("At least one fault type must be specified")

        self.fault_types = list(fault_types)
        self.probability = probability
        self.after_step = after_step
        self.on_fault = on_fault  # callback(step, rank, fault_type) before injection

        # Weights default to uniform
        if weights is not None:
            if len(weights) != len(fault_types):
                raise ValueError(
                    f"weights length ({len(weights)}) must match "
                    f"fault_types length ({len(fault_types)})"
                )
            self.weights = list(weights)
        else:
            self.weights = [1.0] * len(fault_types)

        # Dispatch table
        self._dispatch = {
            "exception": self._inject_exception,
            "sigkill": self._inject_sigkill,
            "hang": self._inject_hang,
        }

        # Stats
        self.injected_count = 0
        self.injected_by_type = {ft: 0 for ft in fault_types}
        self.injected_steps: Set[int] = set()  # Track already-injected steps

        # Pre-populate injected steps from prior container runs (for baseline
        # mode where FaultInjector is recreated on each container restart).
        # Without this, deterministic faults re-trigger on the same step
        # after container restart, creating an infinite crash loop.
        if pre_injected_steps:
            self.injected_steps.update(pre_injected_steps)

        # Fault timing (for shutdown time measurement)
        self.last_fault_time: Optional[float] = None  # time.time() of last injection
        self.last_fault_step: Optional[int] = None  # step of last injection

        # Shared store for broadcasting fault time to all ranks (Option D).
        # The faulting rank writes fault_time to the store; all ranks read it
        # at function re-entry to compute shutdown time. Works across ranks
        # because the store (TCPStore/PrefixStore) is shared.
        self._shared_store = shared_store

        # Deterministic fault schedule (if fault_count is set)
        self.fault_count = fault_count
        self.fault_schedule: Dict[
            int, Tuple[int, str]
        ] = {}  # step -> (target_rank, fault_type)

        if fault_count is not None:
            self._generate_schedule(
                fault_count, fault_seed, max_steps, world_size, after_step
            )

    def _generate_schedule(
        self,
        fault_count: int,
        seed: int,
        max_steps: int,
        world_size: int,
        after_step: int,
    ) -> None:
        """Pre-generate a deterministic fault schedule.

        Creates exactly ``fault_count`` faults at unique random steps in
        [after_step+1, max_steps], each targeting a random non-zero rank
        with a pre-determined fault type. The schedule is identical across
        all ranks (same seed), making experiments fully reproducible.
        """
        rng = random.Random(seed)
        eligible_steps = list(range(after_step + 1, max_steps + 1))
        if fault_count > len(eligible_steps):
            fault_count = len(eligible_steps)
        fault_steps = sorted(rng.sample(eligible_steps, fault_count))

        # Assign each fault a target rank and fault type
        for step in fault_steps:
            target_rank = rng.randint(1, world_size - 1)
            fault_type = rng.choices(self.fault_types, weights=self.weights, k=1)[0]
            self.fault_schedule[step] = (target_rank, fault_type)

        logger.info(
            f"Deterministic fault schedule (seed={seed}): "
            f"{[(s, r, t) for s, (r, t) in sorted(self.fault_schedule.items())]}"
        )

    def maybe_inject(self, step: int, rank: int) -> None:
        """
        Potentially inject a fault at the given training step.

        In deterministic mode, injects only if this step/rank matches the
        pre-generated schedule. In stochastic mode, rolls against probability.

        Only injects on non-zero ranks and after ``after_step`` warmup.
        Call this once per training step.

        Args:
            step: Current training step number.
            rank: Current distributed rank.
        """
        # Never inject on rank 0 (rendezvous host)
        if rank == 0:
            return

        # Warmup period
        if step <= self.after_step:
            return

        if self.fault_count is not None:
            # Deterministic mode: check pre-generated schedule
            if step not in self.fault_schedule:
                return
            # Skip faults that were already injected (e.g., after in-process
            # restart rolls back to a step before this fault)
            if step in self.injected_steps:
                return
            target_rank, scheduled_fault_type = self.fault_schedule[step]
            # Record fault time on ALL ranks (for shutdown measurement).
            # Only the target rank actually injects, but all ranks will
            # experience the restart and need to know when the fault occurred.
            self.last_fault_time = time.time()
            self.last_fault_step = step
            # Broadcast fault time via shared store so rank 0 (which writes
            # results JSON) can compute accurate shutdown time even if it
            # didn't reach this step before the Wrapper interrupted it.
            if self._shared_store is not None:
                try:
                    self._shared_store.set("last_fault_time", str(self.last_fault_time))
                    self._shared_store.set("last_fault_step", str(step))
                except Exception:
                    pass  # Store write failure is non-critical
            if target_rank != rank:
                return
        else:
            # Stochastic mode: probabilistic trigger
            if random.random() >= self.probability:
                return
            scheduled_fault_type = None

        # Choose fault type: use pre-generated type in deterministic mode,
        # weighted random in stochastic mode
        if scheduled_fault_type is not None:
            fault_type = scheduled_fault_type
        else:
            fault_type = random.choices(self.fault_types, weights=self.weights, k=1)[0]

        self.injected_count += 1
        self.injected_by_type[fault_type] += 1
        self.injected_steps.add(step)

        logger.warning(
            f"INJECTING FAULT at step {step}! "
            f"type={fault_type}, "
            f"fault #{self.injected_count}"
        )

        # Pre-injection callback (e.g. record fault step in metrics).
        # Called before dispatch because SIGKILL kills the process instantly.
        if self.on_fault is not None:
            self.on_fault(step, rank, fault_type)

        self._dispatch[fault_type](step, rank)

    def _inject_exception(self, step: int, rank: int) -> None:
        """Raise a RuntimeError. Recoverable by in-process restart."""
        raise RuntimeError(f"Simulated fault (exception) at step {step} on rank {rank}")

    def _inject_sigkill(self, step: int, rank: int) -> None:
        """Send SIGKILL to this process. Only recoverable by ft_launcher or K8s restart."""
        # Flush logs so the warning message above is visible
        for handler in logging.getLogger().handlers:
            handler.flush()
        os.kill(os.getpid(), signal.SIGKILL)

    def _inject_hang(self, step: int, rank: int) -> None:
        """Simulate a hung process (e.g., NCCL deadlock, GPU stall).

        Uses time.sleep() to keep the process alive but unresponsive.
        Detection depends on the fault tolerance mechanism:
        - NVRx inprocess.Wrapper: soft_timeout detects unresponsive rank
        - ft_launcher: heartbeat timeout detects hung worker
        - Baseline: NCCL timeout causes all ranks to crash

        NOTE: Hang faults with inprocess.Wrapper may cause cascading
        restarts on the hung rank's node because the Wrapper cannot kill
        the sleeping thread. Use exception-only faults for in-process
        restart experiments. Hang faults work correctly with ft_launcher
        and baseline (K8s restart) modes.
        """
        while True:
            time.sleep(3600)

    def summary(self) -> str:
        """Return a human-readable summary of injected faults."""
        parts = [f"Total faults injected: {self.injected_count}"]
        for ft in self.fault_types:
            parts.append(f"  {ft}: {self.injected_by_type[ft]}")
        return "\n".join(parts)

    def __repr__(self) -> str:
        if self.fault_count is not None:
            schedule_str = [
                (s, r, t) for s, (r, t) in sorted(self.fault_schedule.items())
            ]
            return (
                f"FaultInjector(mode=deterministic, fault_types={self.fault_types}, "
                f"fault_count={self.fault_count}, "
                f"schedule={schedule_str}, "
                f"after_step={self.after_step})"
            )
        return (
            f"FaultInjector(mode=stochastic, fault_types={self.fault_types}, "
            f"weights={self.weights}, "
            f"probability={self.probability}, "
            f"after_step={self.after_step})"
        )
