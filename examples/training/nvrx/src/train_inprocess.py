#!/usr/bin/env python3
"""
NVRx In-Process Restart Test Script

Demonstrates NVRx in-process restart with FSDP/DDP training.
Compares recovery time and training continuity between:
  - Baseline: No fault tolerance (crash = full restart)
  - NVRx: In-process restart (crash = seconds-level recovery)

The script wraps the training function with nvidia_resiliency_ext.inprocess.Wrapper,
which automatically detects failures and restarts the training on healthy ranks
without restarting the container or re-creating the CUDA context.

Supports both FSDP and DDP via --parallel_strategy flag.

Launch with torchrun for multi-GPU per node:
  torchrun --nnodes=2 --nproc_per_node=8 \\
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \\
    train_inprocess.py [args]

Port allocation:
  MASTER_PORT (29500)     -- torchrun rendezvous (only used by torchrun, not by this script)
  MASTER_PORT + 1         -- inprocess.Wrapper internal store
  MASTER_PORT + 2         -- base TCPStore for NCCL process group (PrefixStore per restart)
"""

import os
import sys
import time
import json
import random
import argparse
import logging
import datetime
from typing import Optional

# Required environment variables for NVRx in-process restart.
# These suppress error rethrows so the Wrapper can handle them.
# In baseline mode (DISABLE_NVRX_WRAPPER=1), we do NOT set these
# so that NCCL errors properly crash the process for K8s to restart.
if os.environ.get("DISABLE_NVRX_WRAPPER", "").lower() not in ("1", "true", "yes"):
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "error")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ.setdefault("TORCH_NCCL_RETHROW_CUDA_ERRORS", "0")

import torch
import torch.distributed as dist

from transformers import AutoTokenizer
from datasets import load_dataset

# Import NVRx in-process restart
try:
    import nvidia_resiliency_ext.inprocess as inprocess
    from nvidia_resiliency_ext.inprocess import CallWrapper

    NVRX_INPROCESS_AVAILABLE = True
except ImportError:
    NVRX_INPROCESS_AVAILABLE = False
    CallWrapper = None

from distributed_utils import (
    add_distributed_args,
    create_model,
    wrap_model,
    create_dataloader as _create_dataloader_shared,
    train_step,
    save_checkpoint as _save_checkpoint_shared,
    load_checkpoint as _load_checkpoint_shared,
)
from metrics_collector import MetricsCollector
from failure_simulator import FaultInjector


# ============================================================================
# Global state that survives across restarts (process-group independent)
# This is a key advantage of in-process restart: we don't re-download
# the dataset or re-tokenize on each recovery.
# ============================================================================
_tokenizer = None
_dataset = None


def str2bool(v):
    """Argparse type for boolean arguments."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def setup_logging(rank):
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="NVRx In-Process Restart Test")

    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=1000)

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
        "Default: exception. Note: sigkill is unrecoverable in pure in-process "
        "mode (will crash the container).",
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

    # In-process restart config
    parser.add_argument(
        "--soft_timeout_seconds",
        type=int,
        default=120,
        help="Soft timeout for hang detection (seconds)",
    )
    parser.add_argument(
        "--hard_timeout_seconds",
        type=int,
        default=180,
        help="Hard timeout for rank termination (seconds)",
    )
    parser.add_argument(
        "--barrier_timeout_seconds",
        type=int,
        default=300,
        help="Barrier timeout (must be > hard_timeout_seconds)",
    )
    parser.add_argument(
        "--max_restarts",
        type=int,
        default=10,
        help="Maximum number of in-process restarts before giving up",
    )
    parser.add_argument(
        "--nccl_timeout_seconds",
        type=int,
        default=300,
        help="NCCL operation timeout (seconds). Lower values detect peer "
        "failures faster in baseline mode.",
    )
    parser.add_argument(
        "--disable_nvrx_wrapper",
        action="store_true",
        default=False,
        help="Run without NVRx in-process restart wrapper (baseline mode). "
        "Faults will crash the container and rely on Kubernetes restart.",
    )

    # Distributed strategy & dtype (from shared utils)
    add_distributed_args(parser)

    return parser.parse_args()


def get_tokenizer(model_name):
    """Get or create tokenizer (survives across restarts)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def get_dataset(dataset_name, streaming=True, dataset_path=None):
    """Get or create dataset (survives across restarts).

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
    """Create dataloader using cached dataset (survives across restarts)."""
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


def save_checkpoint(
    model, optimizer, step, checkpoint_path, rank, strategy, call_wrapper=None
):
    """Save checkpoint, optionally protected by in-process restart atomic context.

    All ranks save (both FSDP and DDP with local emptyDir storage).
    """
    return _save_checkpoint_shared(
        model,
        optimizer,
        step,
        checkpoint_path,
        rank,
        strategy,
        call_wrapper=call_wrapper,
    )


def load_checkpoint(checkpoint_path, model, optimizer, rank, strategy):
    """Load checkpoint if it exists. Returns start_step."""
    start_step, _ = _load_checkpoint_shared(
        checkpoint_path, model, optimizer, rank, strategy
    )
    return start_step


# ============================================================================
# Persistent metrics (survives across K8s container restarts)
# ============================================================================
METRICS_FILENAME = "cumulative_metrics.json"


def load_persistent_metrics(checkpoint_path):
    """Load cumulative metrics from disk, if they exist from a prior container restart."""
    path = os.path.join(checkpoint_path, METRICS_FILENAME)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_persistent_metrics(checkpoint_path, metrics):
    """Save cumulative metrics to disk so they survive container restarts."""
    os.makedirs(checkpoint_path, exist_ok=True)
    path = os.path.join(checkpoint_path, METRICS_FILENAME)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def merge_persistent_metrics(restart_metrics, checkpoint_path):
    """
    Load previously persisted metrics and merge into restart_metrics.

    In baseline mode, each container restart gets a fresh restart_metrics dict.
    This function restores cumulative state (total faults, total steps across
    all container lifetimes, first-ever start time) from the metrics file.
    """
    prior = load_persistent_metrics(checkpoint_path)
    if prior is None:
        # First container start - record the absolute start time
        restart_metrics["job_start_time"] = time.time()
        return

    # Merge cumulative fields from prior container runs
    restart_metrics["container_restarts"] = prior.get("container_restarts", 0) + 1
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
    restart_metrics["injected_fault_steps"] = prior.get("injected_fault_steps", [])


def persist_current_metrics(restart_metrics, checkpoint_path):
    """Persist the current cumulative metrics snapshot to disk."""
    cumulative = {
        "container_restarts": restart_metrics.get("container_restarts", 0),
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
        "injected_fault_steps": restart_metrics.get("injected_fault_steps", []),
    }
    save_persistent_metrics(checkpoint_path, cumulative)


# ============================================================================
# Training function wrapped by NVRx in-process restart
# ============================================================================
def train_with_inprocess_restart(
    args,
    restart_metrics,
    base_store=None,
    fault_injector: Optional[FaultInjector] = None,
    call_wrapper: Optional[CallWrapper] = None,
):
    """
    Training function that can be wrapped by NVRx in-process restart.

    On each invocation (initial or after restart):
    1. Re-initialize the distributed process group
    2. Re-create the FSDP model
    3. Load from latest checkpoint
    4. Resume training from the checkpoint step

    Args:
        base_store: Shared TCPStore for NCCL process group. In NVRx mode, a
            PrefixStore keyed by the restart iteration is created on top of
            this to avoid stale keys. In baseline mode, this is None and a
            fresh TCPStore is created each time (only one iteration).
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    logger = setup_logging(rank)

    # Track restart iteration
    iteration = call_wrapper.iteration if call_wrapper else 0
    restart_entry_time = time.time()

    # Load cumulative metrics from prior container restarts (baseline mode)
    # or prior Wrapper iterations (NVRx mode, iteration 0 only).
    # This must happen before we log anything about restarts.
    if iteration == 0:
        merge_persistent_metrics(restart_metrics, args.checkpoint_path)

    is_container_restart = (
        iteration == 0 and restart_metrics.get("container_restarts", 0) > 0
    )

    if iteration > 0:
        restart_metrics["restart_count"] += 1
        logger.info(f"=" * 60)
        logger.info(f"IN-PROCESS RESTART #{iteration}")
        logger.info(f"World size: {world_size}")
        logger.info(f"=" * 60)
    elif is_container_restart:
        n = restart_metrics["container_restarts"]
        logger.info(f"=" * 60)
        logger.info(f"CONTAINER RESTART #{n} (Kubernetes)")
        logger.info(f"World size: {world_size}")
        logger.info(f"=" * 60)
    else:
        logger.info("=" * 80)
        logger.info("NVRx In-Process Restart Training Starting")
        logger.info("=" * 80)
        logger.info(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
        logger.info(f"In-Process Restart: {'Enabled' if call_wrapper else 'Disabled'}")
        logger.info(f"Fault Injection: {args.inject_faults}")

    # Initialize distributed (must be done each restart)
    torch.cuda.set_device(local_rank)
    t_nccl_start = time.time()
    if world_size > 1:
        if dist.is_initialized():
            dist.destroy_process_group()
            # Allow EFA/libfabric to fully release RDMA connections before
            # re-creating the process group. Without this delay, stale EFA
            # connection state causes transient NCCL errors on re-init,
            # especially at scale (32 EFA devices on p5.48xlarge).
            if iteration > 0:
                time.sleep(5)
                logger.info("EFA cooldown (5s) after process group destroy")

        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = int(os.environ.get("MASTER_PORT", 29500))

        if base_store is not None:
            # NVRx mode: use PrefixStore on the shared base_store, keyed by
            # the restart iteration to avoid stale keys from prior process
            # groups. This matches the pattern in train_ft_launcher.py.
            store = dist.PrefixStore(str(iteration), base_store)
        else:
            # Baseline mode (or single iteration): create a fresh TCPStore.
            # Use MASTER_PORT + 2 to avoid conflict with the Wrapper's
            # internal store on MASTER_PORT + 1.
            store = dist.TCPStore(
                host_name=master_addr,
                port=master_port + 2,
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

    # Create model and wrap with FSDP/DDP
    t_model_start = time.time()
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = get_tokenizer(args.model_name)

    model, _ = create_model(args.model_name, args.torch_dtype)
    model = wrap_model(model, args.parallel_strategy, local_rank, args.model_name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    t_model_end = time.time()

    # Load from checkpoint (resume after fault)
    t_ckpt_load_start = time.time()
    start_step = load_checkpoint(
        args.checkpoint_path, model, optimizer, rank, args.parallel_strategy
    )
    t_ckpt_load_end = time.time()

    # Mark scheduled faults as already injected to prevent re-triggering.
    # Any fault step <= max(start_step, cumulative_steps_completed) was
    # already reached in a prior iteration/container. We use the persisted
    # cumulative_steps_completed because start_step may be 0 if no checkpoint
    # exists yet but faults already fired (e.g., fault at step 62, no ckpt).
    if fault_injector is not None and hasattr(fault_injector, "fault_schedule"):
        max_completed = max(
            start_step,
            restart_metrics.get("cumulative_steps_completed", 0),
        )
        for fstep in fault_injector.fault_schedule:
            if fstep <= max_completed:
                fault_injector.injected_steps.add(fstep)
        # Also load from persisted metrics (for faults before any checkpoint)
        for fstep in restart_metrics.get("injected_fault_steps", []):
            fault_injector.injected_steps.add(fstep)

    # Create dataloader
    t_dl_start = time.time()
    dataloader = create_dataloader(args, tokenizer, rank, world_size)
    dataloader_iter = iter(dataloader)
    t_dl_end = time.time()

    # Training loop
    model.train()
    training_start_time = restart_metrics.get("training_start_time", time.time())
    if "training_start_time" not in restart_metrics:
        restart_metrics["training_start_time"] = training_start_time

    # Use job_start_time for total wall clock (persists across container restarts)
    job_start_time = restart_metrics.get("job_start_time", training_start_time)

    # Measure recovery overhead with per-phase breakdown
    if iteration > 0 or is_container_restart:
        startup_time = time.time() - restart_entry_time
        nccl_time = t_nccl_end - t_nccl_start
        model_time = t_model_end - t_model_start
        ckpt_load_time = t_ckpt_load_end - t_ckpt_load_start
        dl_time = t_dl_end - t_dl_start
    else:
        # First iteration: record setup time (NCCL + model + ckpt + dataloader)
        restart_metrics["setup_time"] = time.time() - restart_entry_time

    if iteration > 0 or is_container_restart:
        startup_time = time.time() - restart_entry_time
        nccl_time = t_nccl_end - t_nccl_start
        model_time = t_model_end - t_model_start
        ckpt_load_time = t_ckpt_load_end - t_ckpt_load_start
        dl_time = t_dl_end - t_dl_start

        # Compute shutdown time: from fault injection to function re-entry.
        # Option D: read from shared store (broadcast by faulting rank).
        # Fallback: read from local FaultInjector or cumulative_metrics.json.
        shutdown_time = 0.0
        if base_store is not None:
            try:
                fault_time_str = base_store.get("last_fault_time")
                shutdown_time = restart_entry_time - float(fault_time_str)
                if shutdown_time < 0:
                    shutdown_time = 0.0  # Clock skew guard
            except Exception:
                pass
        if (
            shutdown_time == 0.0
            and iteration > 0
            and fault_injector
            and fault_injector.last_fault_time
        ):
            shutdown_time = restart_entry_time - fault_injector.last_fault_time
        elif (
            shutdown_time == 0.0
            and is_container_restart
            and "last_fault_time" in restart_metrics
        ):
            persisted_ft = restart_metrics.get("last_fault_time")
            if persisted_ft is not None:
                shutdown_time = restart_entry_time - persisted_ft

        total_recovery = shutdown_time + startup_time
        restart_metrics["recovery_times"].append(total_recovery)

        # Store per-phase breakdown for detailed analysis
        if "recovery_breakdown" not in restart_metrics:
            restart_metrics["recovery_breakdown"] = []
        restart_metrics["recovery_breakdown"].append(
            {
                "iteration": iteration,
                "type": "in-process" if iteration > 0 else "container_restart",
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

        restart_type = "in-process" if iteration > 0 else "container restart"
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

    max_training_time = args.training_duration_minutes * 60
    step = start_step
    checkpoint_count = restart_metrics.get("checkpoint_count", 0)
    total_checkpoint_time = restart_metrics.get("total_checkpoint_time", 0)
    termination_reason = "max_steps_reached"

    logger.info(f"Starting training from step {start_step}...")

    while step < args.max_steps:
        # Check training time limit (wall clock from first start)
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

        # Report progress to in-process restart monitor
        if call_wrapper is not None:
            call_wrapper.ping()

        # Log progress
        if step % 10 == 0:
            logger.info(
                f"Step {step}, Loss: {loss:.4f}, "
                f"Wall: {wall_elapsed:.0f}s, "
                f"Restarts: {restart_metrics['restart_count']}"
            )

        # Fault injection (configurable types: exception, sigkill, hang)
        if fault_injector is not None:
            fault_injector.maybe_inject(step, rank)

        # Checkpoint saving (all ranks must participate - FSDP state_dict is collective)
        if step % args.checkpoint_interval == 0:
            logger.info(f"Saving checkpoint at step {step}...")
            _, ckpt_time = save_checkpoint(
                model,
                optimizer,
                step,
                args.checkpoint_path,
                rank,
                args.parallel_strategy,
                call_wrapper,
            )
            logger.info(f"Checkpoint saved in {ckpt_time:.3f}s")
            total_checkpoint_time += ckpt_time
            checkpoint_count += 1

            # Persist cumulative metrics alongside checkpoint so they survive
            # container crashes in baseline mode.
            restart_metrics["total_steps"] = step
            restart_metrics["total_wall_time"] = time.time() - training_start_time
            restart_metrics["total_checkpoint_time"] = total_checkpoint_time
            restart_metrics["checkpoint_count"] = checkpoint_count
            persist_current_metrics(restart_metrics, args.checkpoint_path)

    # Training complete
    total_wall_time = time.time() - training_start_time
    total_job_time = time.time() - job_start_time
    restart_metrics["total_steps"] = step
    restart_metrics["total_wall_time"] = total_wall_time
    restart_metrics["total_checkpoint_time"] = total_checkpoint_time
    restart_metrics["checkpoint_count"] = checkpoint_count

    # Persist cumulative metrics to disk (survives container restarts)
    persist_current_metrics(restart_metrics, args.checkpoint_path)

    if dist.is_initialized():
        dist.destroy_process_group()

    # Build cumulative totals for summary
    all_fault_steps = restart_metrics.get(
        "cumulative_fault_steps", []
    ) + restart_metrics.get("fault_steps", [])
    all_recovery_times = restart_metrics.get(
        "cumulative_recovery_times", []
    ) + restart_metrics.get("recovery_times", [])
    total_container_restarts = restart_metrics.get("container_restarts", 0)
    in_process_restarts = restart_metrics.get("restart_count", 0)
    cum_ckpt_count = (
        restart_metrics.get("cumulative_checkpoint_count", 0) + checkpoint_count
    )
    cum_ckpt_time = (
        restart_metrics.get("cumulative_checkpoint_time", 0) + total_checkpoint_time
    )

    # Print summary (only on rank 0 or if single process)
    if rank == 0:
        mode_str = (
            "BASELINE (no fault tolerance)"
            if call_wrapper is None
            else "NVRx IN-PROCESS RESTART"
        )
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
        # Cumulative active training across all container lifetimes
        cum_training_time = (
            restart_metrics.get("cumulative_training_time", 0) + total_wall_time
        )
        logger.info(f"  Cumulative active training time: {cum_training_time:.1f}s")
        logger.info(f"  Effective throughput: {step / total_job_time:.2f} steps/sec")
        logger.info("")
        logger.info("RESILIENCY METRICS:")
        logger.info(f"  Container restarts (K8s): {total_container_restarts}")
        logger.info(f"  In-process restarts (NVRx): {in_process_restarts}")
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
            # Steps lost per fault = steps since last checkpoint
            if all_fault_steps:
                steps_lost = [fs % args.checkpoint_interval for fs in all_fault_steps]
                avg_lost = sum(steps_lost) / len(steps_lost)
                logger.info(f"  Avg steps lost per fault: {avg_lost:.0f}")
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
            "container_restarts": total_container_restarts,
            "inprocess_restarts": in_process_restarts,
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

        # Wasted steps: steps computed between last checkpoint and fault,
        # then discarded when rolling back to checkpoint.
        # Use the deterministic fault schedule (all ranks' faults) rather than
        # just rank 0's observed faults, since rank 0 may not inject any faults.
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

        # FSDP/NCCL initial setup time (first container only)
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

        # Infra goodput: (wall_time - fault_downtime) / wall_time
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


def main():
    args = parse_args()

    # Check for disable flag (CLI arg or env var)
    disable_wrapper = args.disable_nvrx_wrapper or os.environ.get(
        "DISABLE_NVRX_WRAPPER", ""
    ).lower() in ("1", "true", "yes")

    # Shared metrics dict (mutable, survives across restarts since it's an arg)
    restart_metrics = {
        "restart_count": 0,
        "recovery_times": [],
        "fault_steps": [],
        "total_steps": 0,
        "total_wall_time": 0,
        "total_checkpoint_time": 0,
        "checkpoint_count": 0,
    }

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
            # Track injected steps for persistence across container restarts
            if "injected_fault_steps" not in restart_metrics:
                restart_metrics["injected_fault_steps"] = []
            restart_metrics["injected_fault_steps"].append(step)
            # Record fault time for shutdown measurement across container restarts
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
            pre_injected_steps=restart_metrics.get("injected_fault_steps", []),
        )
        print(f"Fault injector: {fault_injector}")
    else:
        print("Fault injection disabled")

    if not NVRX_INPROCESS_AVAILABLE or disable_wrapper:
        mode = "BASELINE (no fault tolerance)"
        if disable_wrapper:
            print(f"Mode: {mode} -- NVRx Wrapper disabled by flag")
        else:
            print(f"Mode: {mode} -- NVRx not available")
        print(
            "Faults will crash this process. "
            "Recovery depends on Kubernetes restarting the container."
        )
        # Baseline: no base_store needed (single iteration, fresh TCPStore)
        train_with_inprocess_restart(
            args, restart_metrics, base_store=None, fault_injector=fault_injector
        )
        return

    # NVRx in-process restart mode
    print("Mode: NVRx IN-PROCESS RESTART")

    # Pre-load tokenizer and dataset before wrapping (they survive across restarts)
    print("Pre-loading tokenizer and dataset (will be reused across restarts)...")
    get_tokenizer(args.model_name)
    get_dataset(args.dataset_name, args.streaming, args.dataset_path)
    print("Pre-loading complete.")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", 29500))

    # Base TCPStore shared across all inprocess restart iterations.
    # Port MASTER_PORT+2 avoids conflict with the Wrapper's internal store
    # on MASTER_PORT+1 and torchrun's rendezvous on MASTER_PORT.
    # A PrefixStore keyed by iteration is created on top of this inside
    # the training function to avoid stale keys.
    base_store = dist.TCPStore(
        host_name=master_addr,
        port=master_port + 2,
        world_size=world_size,
        is_master=(rank == 0),
        multi_tenant=True,
        wait_for_workers=True,
        use_libuv=True,
    )

    # Connect fault injector to shared store for fault time broadcast
    if fault_injector is not None:
        fault_injector._shared_store = base_store

    # Wrap training with NVRx in-process restart
    # store_kwargs must include host_name for cross-node connectivity.
    wrapped_train = inprocess.Wrapper(
        store_kwargs={"host_name": master_addr, "port": master_port + 1},
        soft_timeout=datetime.timedelta(seconds=args.soft_timeout_seconds),
        hard_timeout=datetime.timedelta(seconds=args.hard_timeout_seconds),
        barrier_timeout=datetime.timedelta(seconds=args.barrier_timeout_seconds),
        completion_timeout=datetime.timedelta(seconds=args.barrier_timeout_seconds),
        health_check=inprocess.Compose(
            inprocess.health_check.CudaHealthCheck(),
            inprocess.health_check.FaultCounter(max_rank_faults=20),
        ),
        initialize=inprocess.initialize.RetryController(
            max_iterations=args.max_restarts,
            min_active_world_size=1,
        ),
        rank_assignment=inprocess.Compose(
            inprocess.rank_assignment.ActivateAllRanks(),
            inprocess.rank_assignment.ShiftRanks(),
        ),
    )(train_with_inprocess_restart)

    # Run the wrapped training, passing the shared base_store
    wrapped_train(args, restart_metrics, base_store, fault_injector)


if __name__ == "__main__":
    main()
