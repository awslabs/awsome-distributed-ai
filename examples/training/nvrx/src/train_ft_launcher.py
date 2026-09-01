#!/usr/bin/env python3
"""
NVRx ft_launcher In-Job Restart Test Script

Demonstrates NVRx fault tolerance using ft_launcher for in-job restart
on EKS with FSDP/DDP training.  Two modes are supported:

  Default (in-job only):
      ft_launcher monitors worker heartbeats.  If a worker crashes or
      hangs, ft_launcher kills all workers and respawns them.  Recovery
      takes ~10-30 s (process respawn + checkpoint reload) but does NOT
      involve Kubernetes container restarts.

  --inprocess (combined in-job + in-process):
      Adds NVRx inprocess.Wrapper on top of ft_launcher.  Soft faults
      (Python exceptions, NCCL errors) are recovered in-process in ~1-2 s.
      Hard faults (SIGKILL, OOM) fall through to ft_launcher for in-job
      restart.  Hang detection is provided by both the Wrapper (soft/hard
      timeout) and ft_launcher (heartbeat timeout).

ft_launcher replaces torchrun as the launcher.  It sets RANK, WORLD_SIZE,
LOCAL_RANK, MASTER_ADDR, MASTER_PORT automatically.

Supports both FSDP and DDP via --parallel_strategy flag.

Port allocation scheme:
    MASTER_PORT     -> ft_launcher rendezvous (c10d TCPStore)
    MASTER_PORT + 1 -> application TCPStore (base store in combined, PG store in injob-only)
    MASTER_PORT + 2 -> inprocess.Wrapper internal store (only in --inprocess mode)
"""

import os
import sys
import time
import json
import argparse
import logging
import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Environment variable setup (must happen before importing torch)
import random

# ---------------------------------------------------------------------------
# When --inprocess is used, we need to suppress NCCL error rethrows so the
# Wrapper can catch them.  In in-job-only mode we let NCCL errors crash the
# worker so ft_launcher can detect the failure.
#
# We check the env var ENABLE_INPROCESS here because argparse hasn't run yet.
_inprocess_enabled = os.environ.get("ENABLE_INPROCESS", "").lower() in (
    "1",
    "true",
    "yes",
)
if _inprocess_enabled:
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "error")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ.setdefault("TORCH_NCCL_RETHROW_CUDA_ERRORS", "0")
else:
    # In-job-only mode: still need NCCL_NVLS_ENABLE=0 for g5 (no NVSwitch)
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

import torch
import torch.distributed as dist

from transformers import AutoTokenizer
from datasets import load_dataset

# NVRx imports
try:
    import nvidia_resiliency_ext.inprocess as inprocess
    from nvidia_resiliency_ext.inprocess import CallWrapper

    NVRX_INPROCESS_AVAILABLE = True
except ImportError:
    NVRX_INPROCESS_AVAILABLE = False
    CallWrapper = None

try:
    import nvidia_resiliency_ext.fault_tolerance as fault_tolerance

    NVRX_FT_AVAILABLE = True
except ImportError:
    NVRX_FT_AVAILABLE = False

from distributed_utils import (
    add_distributed_args,
    create_model,
    wrap_model,
    train_step,
    save_checkpoint as _save_checkpoint_shared,
    load_checkpoint as _load_checkpoint_shared,
)
from failure_simulator import FaultInjector

# ============================================================================
# Global state that survives across in-process restarts
# ============================================================================
_tokenizer = None
_dataset = None


def setup_logging(rank):
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger(__name__)


def str2bool(v):
    """Argparse type for boolean arguments."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="NVRx ft_launcher In-Job Restart Test")

    # Model / training
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=1000)

    # Dataset
    parser.add_argument("--dataset_name", type=str, default="allenai/c4")
    parser.add_argument("--streaming", type=str2bool, default=True)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to pre-downloaded dataset (created by prepare_dataset.py). "
        "If set, loads from local disk with zero HF API calls. "
        "If not set, streams from HuggingFace Hub.",
    )

    # Checkpointing
    parser.add_argument("--checkpoint_path", type=str, default="/checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=50)

    # Training duration
    parser.add_argument("--training_duration_minutes", type=int, default=10)

    # Fault injection
    parser.add_argument(
        "--inject_faults",
        action="store_true",
        default=False,
        help="Enable random fault injection for testing",
    )
    parser.add_argument(
        "--fault_probability",
        type=float,
        default=0.002,
        help="Probability of fault per step (only on rank 1)",
    )
    parser.add_argument(
        "--fault_after_step",
        type=int,
        default=50,
        help="Only inject faults after this step (allow initial training)",
    )
    parser.add_argument(
        "--fault_types",
        type=str,
        default="exception",
        help="Comma-separated fault types to inject: exception, sigkill, hang. "
        "Default: exception. For combined mode use 'exception,sigkill' to "
        "exercise both in-process and in-job restart paths.",
    )
    parser.add_argument(
        "--fault_type_weights",
        type=str,
        default=None,
        help="Comma-separated weights for fault types (same length as --fault_types). "
        "Default: uniform. Example: '0.7,0.3' for 70%% exception / 30%% sigkill.",
    )
    parser.add_argument(
        "--fault_count",
        type=int,
        default=None,
        help="If set, inject exactly this many faults at pre-determined steps "
        "(deterministic mode). Overrides --fault_probability. Use with "
        "--fault_seed for reproducible fault patterns across experiments.",
    )
    parser.add_argument(
        "--fault_seed",
        type=int,
        default=42,
        help="Random seed for deterministic fault schedule. Only used when "
        "--fault_count is set. Same seed = same fault steps and target ranks.",
    )

    # NCCL timeout
    parser.add_argument(
        "--nccl_timeout_seconds",
        type=int,
        default=60,
        help="NCCL operation timeout (seconds).",
    )

    # In-process restart (opt-in)
    parser.add_argument(
        "--inprocess",
        action="store_true",
        default=False,
        help="Enable NVRx in-process restart on top of ft_launcher in-job restart. "
        "Without this flag, only ft_launcher in-job restart is used.",
    )
    parser.add_argument(
        "--soft_timeout_seconds",
        type=int,
        default=120,
        help="Soft timeout for in-process hang detection (seconds)",
    )
    parser.add_argument(
        "--hard_timeout_seconds",
        type=int,
        default=180,
        help="Hard timeout for in-process rank termination (seconds)",
    )
    parser.add_argument(
        "--barrier_timeout_seconds",
        type=int,
        default=300,
        help="Barrier/completion timeout for in-process Wrapper (seconds). "
        "Must be > hard_timeout_seconds.",
    )
    parser.add_argument(
        "--max_inprocess_restarts",
        type=int,
        default=10,
        help="Maximum number of in-process restarts before giving up",
    )

    # Distributed strategy & dtype (from shared utils)
    add_distributed_args(parser)

    return parser.parse_args()


# ============================================================================
# Data loading (global state survives in-process restarts)
# ============================================================================
def get_tokenizer(model_name):
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def get_dataset(dataset_name, streaming=True, dataset_path=None):
    """Get or create dataset.

    If dataset_path is set, loads a pre-downloaded dataset from local disk
    (no HuggingFace API calls -- eliminates 429 rate limiting on restarts).
    Otherwise, streams from HuggingFace Hub with retry logic.

    Use prepare_dataset.py to create the local dataset:
        python prepare_dataset.py --output_path /checkpoints/c4_subset
    """
    global _dataset
    if _dataset is None:
        if dataset_path is not None:
            from datasets import load_from_disk

            _dataset = load_from_disk(dataset_path)
            print(
                f"Loaded dataset from local path: {dataset_path} ({len(_dataset)} samples)"
            )
        else:
            # Retry with exponential backoff for HuggingFace 429 rate limiting.
            for attempt in range(10):
                try:
                    _dataset = load_dataset(
                        dataset_name,
                        "en",
                        split="train",
                        streaming=streaming,
                        trust_remote_code=True,
                    )
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 9:
                        wait = 2**attempt + random.random() * 2
                        logging.getLogger(__name__).warning(
                            f"HF rate limit (429), retry {attempt + 1}/10 in {wait:.1f}s"
                        )
                        time.sleep(wait)
                    else:
                        raise
    return _dataset


def create_dataloader(args, tokenizer, rank, world_size):
    dataset = get_dataset(args.dataset_name, args.streaming, args.dataset_path)

    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_seq_length,
            padding=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": encodings["input_ids"].clone(),
        }

    if world_size > 1:
        dataset_shard = dataset.shard(num_shards=world_size, index=rank)
    else:
        dataset_shard = dataset

    from torch.utils.data import DataLoader

    return DataLoader(
        dataset_shard,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=2,
    )


# train_step is imported from distributed_utils


# ============================================================================
# Checkpoint save / load (delegates to shared utils, adds ft_client state)
# ============================================================================
def save_checkpoint(
    model,
    optimizer,
    step,
    checkpoint_path,
    rank,
    strategy,
    ft_client=None,
    call_wrapper=None,
):
    """Save checkpoint with optional ft_client state."""
    extra_state = {}
    if ft_client is not None and ft_client.is_initialized:
        extra_state["ft_state"] = ft_client.state_dict()

    return _save_checkpoint_shared(
        model,
        optimizer,
        step,
        checkpoint_path,
        rank,
        strategy,
        call_wrapper=call_wrapper,
        extra_state=extra_state if extra_state else None,
    )


def load_checkpoint(checkpoint_path, model, optimizer, rank, strategy, ft_client=None):
    """Load checkpoint with optional ft_client state restore."""
    extra_loaders = {}
    if ft_client is not None:
        extra_loaders["ft_state"] = lambda state: ft_client.load_state_dict(state)

    start_step, _ = _load_checkpoint_shared(
        checkpoint_path,
        model,
        optimizer,
        rank,
        strategy,
        extra_loaders=extra_loaders if extra_loaders else None,
    )
    return start_step


# ============================================================================
# Persistent metrics (survives K8s container restarts and ft_launcher respawns)
# ============================================================================
METRICS_FILENAME = "cumulative_metrics.json"


def load_persistent_metrics(checkpoint_path):
    path = os.path.join(checkpoint_path, METRICS_FILENAME)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_persistent_metrics(checkpoint_path, metrics):
    os.makedirs(checkpoint_path, exist_ok=True)
    path = os.path.join(checkpoint_path, METRICS_FILENAME)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def merge_persistent_metrics(restart_metrics, checkpoint_path):
    """Merge metrics from prior container / in-job restarts."""
    prior = load_persistent_metrics(checkpoint_path)
    if prior is None:
        restart_metrics["job_start_time"] = time.time()
        return

    restart_metrics["injob_restarts"] = prior.get("injob_restarts", 0) + 1
    restart_metrics["cumulative_fault_steps"] = prior.get("cumulative_fault_steps", [])
    restart_metrics["cumulative_recovery_times"] = prior.get(
        "cumulative_recovery_times", []
    )
    restart_metrics["cumulative_steps_completed"] = prior.get(
        "cumulative_steps_completed", 0
    )
    restart_metrics["cumulative_training_time"] = prior.get(
        "cumulative_training_time", 0
    )
    restart_metrics["cumulative_checkpoint_time"] = prior.get(
        "cumulative_checkpoint_time", 0
    )
    restart_metrics["cumulative_checkpoint_count"] = prior.get(
        "cumulative_checkpoint_count", 0
    )
    restart_metrics["job_start_time"] = prior.get("job_start_time", time.time())
    restart_metrics["recovery_breakdown"] = prior.get("recovery_breakdown", [])
    restart_metrics["last_fault_time"] = prior.get("last_fault_time")


def persist_current_metrics(restart_metrics, checkpoint_path):
    cumulative = {
        "injob_restarts": restart_metrics.get("injob_restarts", 0),
        "cumulative_fault_steps": restart_metrics.get("cumulative_fault_steps", [])
        + restart_metrics.get("fault_steps", []),
        "cumulative_recovery_times": restart_metrics.get(
            "cumulative_recovery_times", []
        )
        + restart_metrics.get("recovery_times", []),
        "cumulative_steps_completed": restart_metrics.get("total_steps", 0),
        "cumulative_training_time": restart_metrics.get("cumulative_training_time", 0)
        + restart_metrics.get("total_wall_time", 0),
        "cumulative_checkpoint_time": restart_metrics.get(
            "cumulative_checkpoint_time", 0
        )
        + restart_metrics.get("total_checkpoint_time", 0),
        "cumulative_checkpoint_count": restart_metrics.get(
            "cumulative_checkpoint_count", 0
        )
        + restart_metrics.get("checkpoint_count", 0),
        "job_start_time": restart_metrics.get("job_start_time", time.time()),
        "recovery_breakdown": restart_metrics.get("recovery_breakdown", []),
        "last_fault_time": restart_metrics.get("last_fault_time"),
    }
    save_persistent_metrics(checkpoint_path, cumulative)


# ============================================================================
# Core training function
#
# In combined mode (--inprocess), this function is wrapped by
# inprocess.Wrapper and may be re-invoked multiple times.
#
# In in-job-only mode, this function runs once per ft_launcher worker
# spawn.  If the worker crashes, ft_launcher respawns a new process that
# calls this function again from scratch.
# ============================================================================
def train_fn(
    args,
    restart_metrics,
    ft_client,
    base_store,
    fault_injector: Optional[FaultInjector] = None,
    call_wrapper: Optional[CallWrapper] = None,
):
    """
    Training function.

    On each invocation (initial, after in-process restart, or after in-job respawn):
    1. Reconnect FT client to rank monitor
    2. Re-initialize the distributed process group
    3. Re-create the FSDP model
    4. Load from latest checkpoint
    5. Resume training
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    logger = setup_logging(rank)

    # Determine restart type and iteration
    iteration = call_wrapper.iteration if call_wrapper else 0
    restart_entry_time = time.time()

    # Load cumulative metrics from prior in-job restarts (iteration 0 only)
    if iteration == 0:
        merge_persistent_metrics(restart_metrics, args.checkpoint_path)

    is_injob_restart = iteration == 0 and restart_metrics.get("injob_restarts", 0) > 0

    if iteration > 0:
        restart_metrics["inprocess_restart_count"] += 1
        logger.info("=" * 60)
        logger.info(f"IN-PROCESS RESTART #{iteration}")
        logger.info(f"World size: {world_size}")
        logger.info("=" * 60)
    elif is_injob_restart:
        n = restart_metrics["injob_restarts"]
        logger.info("=" * 60)
        logger.info(f"IN-JOB RESTART #{n} (ft_launcher respawn)")
        logger.info(f"World size: {world_size}")
        logger.info("=" * 60)
    else:
        mode_str = (
            "COMBINED (ft_launcher + inprocess)"
            if call_wrapper
            else "IN-JOB ONLY (ft_launcher)"
        )
        logger.info("=" * 80)
        logger.info(f"ft_launcher Training Starting - {mode_str}")
        logger.info("=" * 80)
        logger.info(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
        logger.info(f"In-Process Restart: {'Enabled' if call_wrapper else 'Disabled'}")
        logger.info(f"Fault Injection: {args.inject_faults}")

    # -----------------------------------------------------------------------
    # Initialize distributed process group
    # -----------------------------------------------------------------------
    torch.cuda.set_device(local_rank)
    t_nccl_start = time.time()
    if dist.is_initialized():
        dist.destroy_process_group()

    # NOTE: Do NOT add an EFA cooldown here for ft_launcher. Unlike
    # in-process restart (same process, stale EFA state), ft_launcher
    # spawns fresh processes. Adding a sleep causes timing mismatches
    # between nodes (workers sleep different durations, miss each other's
    # NCCL init window, cascade restarts). ft_launcher's own rendezvous
    # handles coordination.

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", 29500))

    # ft_launcher sets MASTER_ADDR to localhost (each pod's ft_launcher
    # manages only local workers).  For the cross-pod NCCL process group
    # we need the actual rank-0 pod IP.  Use PG_MASTER_ADDR if set by
    # the K8s manifest; fall back to MASTER_ADDR (works for single-node).
    pg_master_addr = os.environ.get("PG_MASTER_ADDR", master_addr)

    if base_store is not None:
        # Combined mode: use PrefixStore on the shared base_store, keyed by
        # the inprocess restart iteration to avoid stale keys.
        store = dist.PrefixStore(str(iteration), base_store)
    else:
        # In-job-only mode: create a TCPStore on MASTER_PORT+1 to avoid
        # conflict with ft_launcher's rendezvous store on MASTER_PORT.
        store = dist.TCPStore(
            host_name=pg_master_addr,
            port=master_port + 1,
            world_size=world_size,
            is_master=(rank == 0),
            multi_tenant=True,
            wait_for_workers=True,
            use_libuv=True,
        )

    dist.init_process_group(
        backend="nccl",
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=args.nccl_timeout_seconds),
    )
    logger.info(f"Distributed initialized: rank={rank}, world_size={world_size}")
    t_nccl_end = time.time()

    # -----------------------------------------------------------------------
    # Initialize FT client AFTER process group (needs distributed context)
    # -----------------------------------------------------------------------
    if ft_client is not None:
        if ft_client.is_initialized:
            ft_client.shutdown_workload_monitoring()
        ft_client.init_workload_monitoring()
        logger.info("FT client: workload monitoring initialized")

    # -----------------------------------------------------------------------
    # Create model and wrap with FSDP
    # -----------------------------------------------------------------------
    t_model_start = time.time()
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = get_tokenizer(args.model_name)

    model, _ = create_model(args.model_name, args.torch_dtype)
    model = wrap_model(model, args.parallel_strategy, local_rank, args.model_name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    t_model_end = time.time()

    # Load from checkpoint
    t_ckpt_load_start = time.time()
    start_step = load_checkpoint(
        args.checkpoint_path, model, optimizer, rank, args.parallel_strategy, ft_client
    )
    t_ckpt_load_end = time.time()

    # Mark scheduled faults as already injected to prevent re-triggering
    # after ft_launcher restart. Same logic as train_inprocess.py.
    if fault_injector is not None and hasattr(fault_injector, "fault_schedule"):
        max_completed = max(
            start_step,
            restart_metrics.get("cumulative_steps_completed", 0),
        )
        for fstep in fault_injector.fault_schedule:
            if fstep <= max_completed:
                fault_injector.injected_steps.add(fstep)

    # Create dataloader
    t_dl_start = time.time()
    dataloader = create_dataloader(args, tokenizer, rank, world_size)
    dataloader_iter = iter(dataloader)
    t_dl_end = time.time()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    model.train()
    training_start_time = restart_metrics.get("training_start_time", time.time())
    if "training_start_time" not in restart_metrics:
        restart_metrics["training_start_time"] = training_start_time

    job_start_time = restart_metrics.get("job_start_time", training_start_time)

    # Recovery overhead measurement with per-phase breakdown
    if iteration > 0 or is_injob_restart:
        startup_time = time.time() - restart_entry_time
        nccl_time = t_nccl_end - t_nccl_start
        model_time = t_model_end - t_model_start
        ckpt_load_time = t_ckpt_load_end - t_ckpt_load_start
        dl_time = t_dl_end - t_dl_start

        # Compute shutdown time.
        # Priority: (1) shared store (in-process mode with base_store),
        # (2) local FaultInjector (same process),
        # (3) persisted last_fault_time from cumulative_metrics.json
        #     (ft_launcher in-job mode: new process, old fault time on FSx).
        # Shutdown includes: ft_launcher heartbeat detection, worker kill,
        # cross-node rendezvous, and new process spawn.
        shutdown_time = 0.0
        if base_store is not None:
            try:
                fault_time_str = base_store.get("last_fault_time")
                shutdown_time = restart_entry_time - float(fault_time_str)
                if shutdown_time < 0:
                    shutdown_time = 0.0
            except Exception:
                pass
        if shutdown_time == 0.0 and fault_injector and fault_injector.last_fault_time:
            shutdown_time = restart_entry_time - fault_injector.last_fault_time
        if shutdown_time == 0.0 and is_injob_restart:
            persisted_ft = restart_metrics.get("last_fault_time")
            if persisted_ft is not None:
                shutdown_time = restart_entry_time - persisted_ft
                if shutdown_time < 0:
                    shutdown_time = 0.0

        total_recovery = shutdown_time + startup_time
        restart_metrics["recovery_times"].append(total_recovery)

        if "recovery_breakdown" not in restart_metrics:
            restart_metrics["recovery_breakdown"] = []
        restart_metrics["recovery_breakdown"].append(
            {
                "iteration": iteration,
                "type": "in-process" if iteration > 0 else "in-job",
                "shutdown_seconds": round(shutdown_time, 3),
                "startup_nccl_seconds": round(nccl_time, 3),
                "startup_model_seconds": round(model_time, 3),
                "startup_ckpt_load_seconds": round(ckpt_load_time, 3),
                "startup_dataloader_seconds": round(dl_time, 3),
                "startup_total_seconds": round(startup_time, 3),
                "total_seconds": round(total_recovery, 3),
                "resumed_from_step": start_step,
            }
        )

        restart_type = "in-process" if iteration > 0 else "in-job (ft_launcher)"
        logger.info(
            f"Recovery overhead [{restart_type}]: "
            f"shutdown={shutdown_time:.3f}s, "
            f"startup={startup_time:.3f}s "
            f"(NCCL: {nccl_time:.3f}s, "
            f"Model: {model_time:.3f}s, "
            f"Ckpt: {ckpt_load_time:.3f}s, "
            f"DL: {dl_time:.3f}s), "
            f"total={total_recovery:.3f}s, "
            f"resumed from step {start_step}"
        )
    else:
        # First iteration: record setup time
        restart_metrics["setup_time"] = time.time() - restart_entry_time

    max_training_time = args.training_duration_minutes * 60
    step = start_step
    checkpoint_count = restart_metrics.get("checkpoint_count", 0)
    total_checkpoint_time = restart_metrics.get("total_checkpoint_time", 0)
    termination_reason = "max_steps_reached"

    logger.info(f"Starting training from step {start_step}...")

    while step < args.max_steps:
        wall_elapsed = time.time() - training_start_time
        if wall_elapsed >= max_training_time:
            logger.info(
                f"Training time limit reached ({args.training_duration_minutes} min)"
            )
            termination_reason = "time_limit"
            break

        # Get batch
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        # Training step
        loss = train_step(model, batch, optimizer)
        step += 1

        # Signal liveness to both monitors
        if call_wrapper is not None:
            call_wrapper.ping()  # in-process Wrapper liveness
        if ft_client is not None and ft_client.is_initialized:
            ft_client.send_heartbeat()  # ft_launcher rank monitor liveness

        # Log progress
        if step % 10 == 0:
            logger.info(
                f"Step {step}, Loss: {loss:.4f}, "
                f"Wall: {wall_elapsed:.0f}s, "
                f"InProcess: {restart_metrics['inprocess_restart_count']}, "
                f"InJob: {restart_metrics.get('injob_restarts', 0)}"
            )

        # Fault injection (configurable types: exception, sigkill, hang)
        if fault_injector is not None:
            fault_injector.maybe_inject(step, rank)

        # Checkpoint saving
        if step % args.checkpoint_interval == 0:
            logger.info(f"Saving checkpoint at step {step}...")
            _, ckpt_time = save_checkpoint(
                model,
                optimizer,
                step,
                args.checkpoint_path,
                rank,
                args.parallel_strategy,
                ft_client=ft_client,
                call_wrapper=call_wrapper,
            )
            logger.info(f"Checkpoint saved in {ckpt_time:.3f}s")
            total_checkpoint_time += ckpt_time
            checkpoint_count += 1

            # Persist metrics alongside checkpoint
            restart_metrics["total_steps"] = step
            restart_metrics["total_wall_time"] = time.time() - training_start_time
            restart_metrics["total_checkpoint_time"] = total_checkpoint_time
            restart_metrics["checkpoint_count"] = checkpoint_count
            persist_current_metrics(restart_metrics, args.checkpoint_path)

    # -----------------------------------------------------------------------
    # Training complete
    # -----------------------------------------------------------------------
    total_wall_time = time.time() - training_start_time
    total_job_time = time.time() - job_start_time
    restart_metrics["total_steps"] = step
    restart_metrics["total_wall_time"] = total_wall_time
    restart_metrics["total_checkpoint_time"] = total_checkpoint_time
    restart_metrics["checkpoint_count"] = checkpoint_count

    persist_current_metrics(restart_metrics, args.checkpoint_path)

    if dist.is_initialized():
        dist.destroy_process_group()

    # -----------------------------------------------------------------------
    # Summary (rank 0 only)
    # -----------------------------------------------------------------------
    all_fault_steps = restart_metrics.get(
        "cumulative_fault_steps", []
    ) + restart_metrics.get("fault_steps", [])
    all_recovery_times = restart_metrics.get(
        "cumulative_recovery_times", []
    ) + restart_metrics.get("recovery_times", [])
    total_injob_restarts = restart_metrics.get("injob_restarts", 0)
    inprocess_restarts = restart_metrics.get("inprocess_restart_count", 0)
    cum_ckpt_count = (
        restart_metrics.get("cumulative_checkpoint_count", 0) + checkpoint_count
    )
    cum_ckpt_time = (
        restart_metrics.get("cumulative_checkpoint_time", 0) + total_checkpoint_time
    )

    if rank == 0:
        if call_wrapper is not None:
            mode_str = "COMBINED (ft_launcher + inprocess)"
        else:
            mode_str = "IN-JOB ONLY (ft_launcher)"

        # Run completion status
        reason_label = (
            "completed" if termination_reason == "max_steps_reached" else "time-limited"
        )

        logger.info("=" * 80)
        logger.info(f"TRAINING SUMMARY - {mode_str}")
        logger.info("=" * 80)
        logger.info(f"Model: {args.model_name}")
        logger.info(f"World Size: {world_size}")
        logger.info(f"Parallel Strategy: {args.parallel_strategy}")
        logger.info(f"Mode: {mode_str}")
        logger.info(f"NCCL Timeout: {args.nccl_timeout_seconds}s")
        logger.info(f"Fault Injection: {args.inject_faults}")
        logger.info("")
        logger.info("TRAINING METRICS:")
        logger.info(f"  Steps: {step}/{args.max_steps} ({reason_label})")
        logger.info(f"  Termination: {termination_reason}")
        logger.info(f"  Job wall time (incl. all restarts): {total_job_time:.1f}s")
        cum_training_time = (
            restart_metrics.get("cumulative_training_time", 0) + total_wall_time
        )
        logger.info(f"  Cumulative active training time: {cum_training_time:.1f}s")
        logger.info(f"  Effective throughput: {step / total_job_time:.2f} steps/sec")
        logger.info("")
        logger.info("RESILIENCY METRICS:")
        logger.info(f"  In-job restarts (ft_launcher): {total_injob_restarts}")
        logger.info(f"  In-process restarts (NVRx Wrapper): {inprocess_restarts}")
        logger.info(f"  Container restarts (K8s): 0 (ft_launcher handles recovery)")
        logger.info(f"  Fault injection steps: {all_fault_steps}")
        if all_recovery_times:
            avg_recovery = sum(all_recovery_times) / len(all_recovery_times)
            max_recovery = max(all_recovery_times)
            min_recovery = min(all_recovery_times)
            total_downtime = total_job_time - cum_training_time
            logger.info(f"  Avg recovery overhead: {avg_recovery:.3f}s")
            logger.info(f"  Min recovery overhead: {min_recovery:.3f}s")
            logger.info(f"  Max recovery overhead: {max_recovery:.3f}s")
            logger.info(
                f"  Total downtime (job time - cumulative training): {total_downtime:.1f}s"
            )
            efficiency = (
                (cum_training_time / total_job_time) * 100
                if total_job_time > 0
                else 100
            )
            logger.info(f"  Training efficiency: {efficiency:.1f}%")
        else:
            logger.info("  No restarts occurred")
        logger.info("")
        logger.info("CHECKPOINT METRICS:")
        logger.info(f"  Total checkpoints: {cum_ckpt_count}")
        logger.info(f"  Checkpoint interval: every {args.checkpoint_interval} steps")
        if cum_ckpt_count > 0:
            logger.info(f"  Avg checkpoint time: {cum_ckpt_time / cum_ckpt_count:.3f}s")
        logger.info("=" * 80)

        # Write results JSON for automated comparison
        results = {
            "experiment_type": "fault_recovery",
            "mode": mode_str,
            "model": args.model_name,
            "world_size": world_size,
            "parallel_strategy": args.parallel_strategy,
            "max_steps": args.max_steps,
            "steps_completed": step,
            "termination_reason": termination_reason,
            "job_wall_time_seconds": round(total_job_time, 1),
            "cumulative_training_time_seconds": round(cum_training_time, 1),
            "effective_throughput_steps_per_sec": round(step / total_job_time, 4)
            if total_job_time > 0
            else 0,
            "training_efficiency_pct": round(
                (cum_training_time / total_job_time) * 100, 2
            )
            if total_job_time > 0
            else 100,
            "injob_restarts": total_injob_restarts,
            "inprocess_restarts": inprocess_restarts,
            "total_faults": len(all_fault_steps),
            "fault_steps": all_fault_steps,
            "recovery_times": [round(t, 3) for t in all_recovery_times],
            "avg_recovery_time_seconds": round(
                sum(all_recovery_times) / len(all_recovery_times), 3
            )
            if all_recovery_times
            else None,
            "total_checkpoints": cum_ckpt_count,
            "checkpoint_interval": args.checkpoint_interval,
            "avg_checkpoint_time_seconds": round(cum_ckpt_time / cum_ckpt_count, 3)
            if cum_ckpt_count > 0
            else None,
            "recovery_breakdown": restart_metrics.get("recovery_breakdown", []),
        }

        # Compute goodput metrics
        recovery_bd = results["recovery_breakdown"]
        total_shutdown = sum(r.get("shutdown_seconds", 0) for r in recovery_bd)
        total_startup = sum(r.get("startup_total_seconds", 0) for r in recovery_bd)
        total_fault_downtime = total_shutdown + total_startup

        wasted_steps = 0
        fault_steps_for_wasted = all_fault_steps
        if fault_injector and fault_injector.fault_count is not None:
            fault_steps_for_wasted = [s for s in fault_injector.fault_schedule.keys()]
        for fs in fault_steps_for_wasted:
            last_ckpt = (fs // args.checkpoint_interval) * args.checkpoint_interval
            wasted_steps += fs - last_ckpt

        # Step time from pure training (excluding checkpoint blocking)
        pure_training_time = cum_training_time - cum_ckpt_time
        avg_step_time = (
            pure_training_time / step if step > 0 and pure_training_time > 0 else 0.37
        )
        wasted_time = wasted_steps * avg_step_time
        setup_time = restart_metrics.get("setup_time", 0)

        # Training goodput: useful training compute / wall time
        # Useful training = completed_steps × avg_step_time (pure compute)
        useful_training_time = step * avg_step_time
        training_goodput = (
            (useful_training_time / total_job_time) * 100 if total_job_time > 0 else 100
        )
        useful_training_time = max(0, useful_training_time)
        training_goodput = (
            (useful_training_time / total_job_time) * 100 if total_job_time > 0 else 100
        )
        infra_goodput = (
            ((total_job_time - total_fault_downtime) / total_job_time) * 100
            if total_job_time > 0
            else 100
        )

        results["training_goodput_pct"] = round(training_goodput, 2)
        results["infra_goodput_pct"] = round(infra_goodput, 2)
        results["wasted_steps"] = wasted_steps
        results["wasted_time_seconds"] = round(wasted_time, 1)
        results["total_shutdown_time_seconds"] = round(total_shutdown, 1)
        results["total_startup_time_seconds"] = round(total_startup, 1)
        results["total_checkpoint_time_seconds"] = round(cum_ckpt_time, 1)
        results["useful_training_time_seconds"] = round(useful_training_time, 1)
        results["setup_time_seconds"] = round(setup_time, 1)
        results["avg_step_time_seconds"] = round(avg_step_time, 4)

        logger.info(f"Training Goodput: {training_goodput:.1f}%")
        logger.info(f"Infra Goodput: {infra_goodput:.1f}%")
        logger.info(
            f"Wasted steps: {wasted_steps} ({wasted_time:.1f}s), "
            f"Shutdown: {total_shutdown:.1f}s, Startup: {total_startup:.1f}s"
        )
        try:
            results_path = os.path.join(args.checkpoint_path, "results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {results_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_port = int(os.environ.get("MASTER_PORT", 29500))

    # Also honour the env var so the K8s manifest can set it
    use_inprocess = args.inprocess or os.environ.get(
        "ENABLE_INPROCESS", ""
    ).lower() in ("1", "true", "yes")

    # ------------------------------------------------------------------
    # Prepare FT client (works in both modes)
    # ------------------------------------------------------------------
    ft_client = None
    if NVRX_FT_AVAILABLE:
        ft_client = fault_tolerance.RankMonitorClient()
        print(f"[Rank {rank}] FT RankMonitorClient created")
    else:
        print(f"[Rank {rank}] WARNING: fault_tolerance not available, no heartbeats")

    # Shared metrics dict
    restart_metrics = {
        "inprocess_restart_count": 0,
        "recovery_times": [],
        "fault_steps": [],
        "total_steps": 0,
        "total_wall_time": 0,
        "total_checkpoint_time": 0,
        "checkpoint_count": 0,
    }

    # Pre-load tokenizer and dataset
    print(f"[Rank {rank}] Pre-loading tokenizer and dataset...")
    get_tokenizer(args.model_name)
    get_dataset(args.dataset_name, args.streaming, args.dataset_path)
    print(f"[Rank {rank}] Pre-loading complete.")

    # ------------------------------------------------------------------
    # Create fault injector (if enabled)
    # ------------------------------------------------------------------
    fault_injector = None
    if args.inject_faults:
        fault_types = [ft.strip() for ft in args.fault_types.split(",")]
        weights = None
        if args.fault_type_weights:
            weights = [float(w.strip()) for w in args.fault_type_weights.split(",")]

        def _on_fault(step, _rank, fault_type):
            restart_metrics["fault_steps"].append(step)
            # Record fault time for shutdown measurement
            restart_metrics["last_fault_time"] = time.time()
            # Persist metrics before injection -- SIGKILL kills process instantly
            restart_metrics["total_steps"] = step
            persist_current_metrics(restart_metrics, args.checkpoint_path)

        fault_injector = FaultInjector(
            fault_types=fault_types,
            weights=weights,
            probability=args.fault_probability,
            after_step=args.fault_after_step,
            on_fault=_on_fault,
            fault_count=args.fault_count,
            fault_seed=args.fault_seed,
            max_steps=args.max_steps,
            world_size=int(os.environ.get("WORLD_SIZE", 2)),
        )
        print(f"[Rank {rank}] Fault injector: {fault_injector}")
    else:
        print(f"[Rank {rank}] Fault injection disabled")

    if use_inprocess:
        # ==============================================================
        # Combined mode: ft_launcher + inprocess.Wrapper
        # ==============================================================
        if not NVRX_INPROCESS_AVAILABLE:
            print("ERROR: --inprocess requested but inprocess module not available")
            sys.exit(1)

        print(f"[Rank {rank}] Mode: COMBINED (ft_launcher + inprocess)")

        # Base TCPStore shared across all inprocess restart iterations.
        # Port MASTER_PORT+1 avoids conflict with ft_launcher's rendezvous
        # on MASTER_PORT.  Use PG_MASTER_ADDR for cross-pod connectivity.
        pg_master_addr = os.environ.get("PG_MASTER_ADDR", os.environ["MASTER_ADDR"])
        base_store = dist.TCPStore(
            host_name=pg_master_addr,
            port=master_port + 1,
            world_size=world_size,
            is_master=(rank == 0),
            multi_tenant=True,
            wait_for_workers=True,
            use_libuv=True,
        )

        # Connect fault injector to shared store for fault time broadcast
        if fault_injector is not None:
            fault_injector._shared_store = base_store

        # Wrap training function with inprocess.Wrapper
        # store_kwargs must include host for cross-pod connectivity since
        # ft_launcher sets MASTER_ADDR=localhost (per-pod).
        wrapped_train = inprocess.Wrapper(
            store_kwargs={"host_name": pg_master_addr, "port": master_port + 2},
            soft_timeout=datetime.timedelta(seconds=args.soft_timeout_seconds),
            hard_timeout=datetime.timedelta(seconds=args.hard_timeout_seconds),
            barrier_timeout=datetime.timedelta(seconds=args.barrier_timeout_seconds),
            completion_timeout=datetime.timedelta(seconds=args.barrier_timeout_seconds),
            health_check=inprocess.Compose(
                inprocess.health_check.CudaHealthCheck(),
                inprocess.health_check.FaultCounter(max_rank_faults=5),
            ),
            initialize=inprocess.initialize.RetryController(
                max_iterations=args.max_inprocess_restarts,
                min_active_world_size=1,
            ),
            rank_assignment=inprocess.Compose(
                inprocess.rank_assignment.ActivateAllRanks(),
                inprocess.rank_assignment.ShiftRanks(),
            ),
        )(train_fn)

        try:
            wrapped_train(args, restart_metrics, ft_client, base_store, fault_injector)
        finally:
            if ft_client is not None and ft_client.is_initialized:
                ft_client.shutdown_workload_monitoring()

    else:
        # ==============================================================
        # In-job only mode: ft_launcher without inprocess.Wrapper
        # ==============================================================
        print(f"[Rank {rank}] Mode: IN-JOB ONLY (ft_launcher)")

        try:
            train_fn(
                args,
                restart_metrics,
                ft_client,
                base_store=None,
                fault_injector=fault_injector,
                call_wrapper=None,
            )
        finally:
            if ft_client is not None and ft_client.is_initialized:
                ft_client.shutdown_workload_monitoring()


if __name__ == "__main__":
    main()
