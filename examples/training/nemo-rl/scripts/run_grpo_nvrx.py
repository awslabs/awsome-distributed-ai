#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""NVRx-enhanced GRPO training wrapper.

This script wraps the standard run_grpo.py to add:
  1. RankMonitorClient heartbeat (connects to ft_launcher's IPC socket)
  2. Nsight profiling env vars (set BEFORE nemo_rl imports)
  3. GPU health check at startup
  4. Delegates to the original run_grpo main()

Invoked by ft_launcher instead of examples/run_grpo.py.
"""

import os
import sys
import signal
import threading
import time

# ── 1. Nsight profiling env vars ──
# Must be set BEFORE importing nemo_rl.utils.nsys (which reads at import time).
# Setting here keeps them in os.environ for the GRPO process only,
# NOT in the K8s pod spec (which would break vLLM's Ray runtime_env).
#
# To enable, uncomment and set desired values:
# os.environ["NRL_NSYS_WORKER_PATTERNS"] = "DTensorPolicyWorkerV2"
# os.environ["NRL_NSYS_PROFILE_STEP_RANGE"] = "3:5"

# ── 2. GPU health check ──
def gpu_health_check():
    """Run basic GPU health validation before training starts."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("[NVRx] WARNING: CUDA not available")
            return False

        gpu_count = torch.cuda.device_count()
        print(f"[NVRx] GPU health check: {gpu_count} GPU(s) detected")

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            # Quick allocation test
            t = torch.zeros(1024, device=f"cuda:{i}")
            del t
            total_gb = props.total_memory / 1e9 if hasattr(props, 'total_memory') else 0
            print(
                f"[NVRx] GPU {i}: {props.name}, "
                f"{total_gb:.1f}GB, "
                f"SM {props.major}.{props.minor}: OK"
            )

        # NVRx CudaHealthCheck if available
        try:
            from nvidia_resiliency_ext.health import CudaHealthCheck
            checker = CudaHealthCheck()
            result = checker.check()
            print(f"[NVRx] CudaHealthCheck: {result}")
        except (ImportError, Exception) as e:
            print(f"[NVRx] CudaHealthCheck not available: {e}")

        print("[NVRx] GPU health: OK")
        return True
    except Exception as e:
        print(f"[NVRx] GPU health check failed: {e}")
        return False


# ── 3. RankMonitorClient heartbeat ──
class HeartbeatThread:
    """Background thread that sends periodic heartbeats to ft_launcher.

    ft_launcher's RankMonitorServer listens on an IPC socket specified by
    the FT_RANK_MONITOR_IPC_SOCKET env var. The RankMonitorClient sends
    heartbeats to indicate the training process is alive.
    """

    def __init__(self, interval_sec=30):
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread = None
        self._client = None

    def start(self):
        """Initialize RankMonitorClient and start heartbeat thread."""
        ipc_socket = os.environ.get("FT_RANK_MONITOR_IPC_SOCKET")
        if not ipc_socket:
            print("[NVRx] FT_RANK_MONITOR_IPC_SOCKET not set — heartbeat disabled")
            print("[NVRx] (This is normal when not running under ft_launcher)")
            return False

        try:
            from nvidia_resiliency_ext.fault_tolerance import RankMonitorClient
            self._client = RankMonitorClient()
            self._client.init_workload_monitoring()
            print(
                f"[NVRx] Heartbeat thread started "
                f"(interval={self.interval_sec}s, socket={ipc_socket})"
            )
        except ImportError:
            print("[NVRx] nvidia_resiliency_ext.fault_tolerance not available — heartbeat disabled")
            return False
        except Exception as e:
            print(f"[NVRx] RankMonitorClient init failed: {e}")
            return False

        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="nvrx-heartbeat",
        )
        self._thread.start()
        return True

    def _heartbeat_loop(self):
        """Send heartbeats at regular intervals until stopped."""
        while not self._stop_event.wait(self.interval_sec):
            try:
                self._client.send_heartbeat()
            except Exception as e:
                print(f"[NVRx] Heartbeat send failed: {e}")

    def stop(self):
        """Stop the heartbeat thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            print("[NVRx] Heartbeat thread stopped")


def main():
    print("=" * 60)
    print("[NVRx] Enhanced GRPO training wrapper")
    print("=" * 60)

    # GPU health check (before heavy imports)
    gpu_health_check()

    # Start heartbeat thread
    heartbeat = HeartbeatThread(interval_sec=30)
    heartbeat.start()

    # Ensure heartbeat stops on signals
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def _signal_handler(signum, frame):
        heartbeat.stop()
        if callable(original_sigterm) and original_sigterm not in (
            signal.SIG_DFL,
            signal.SIG_IGN,
        ):
            original_sigterm(signum, frame)
        else:
            sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _signal_handler)

    # Delegate to the original run_grpo main
    try:
        print("[NVRx] Importing run_grpo.main()...")

        # Ensure /opt/nemo-rl is on sys.path so examples/ imports work
        nemo_rl_root = "/opt/nemo-rl"
        if nemo_rl_root not in sys.path:
            sys.path.insert(0, nemo_rl_root)

        # Also add examples/ to path for direct module resolution
        examples_dir = os.path.join(nemo_rl_root, "examples")
        if examples_dir not in sys.path:
            sys.path.insert(0, examples_dir)

        from examples.run_grpo import main as grpo_main

        print("[NVRx] Starting GRPO training")
        grpo_main()
        print("[NVRx] GRPO training completed successfully")
    except SystemExit as e:
        print(f"[NVRx] GRPO exited with code {e.code}")
        raise
    except Exception as e:
        print(f"[NVRx] GRPO training failed: {e}")
        raise
    finally:
        heartbeat.stop()


if __name__ == "__main__":
    main()
