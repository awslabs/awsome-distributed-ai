#!/usr/bin/env python3
"""
NVRx Local Checkpointing Test Script

Compares NVRx local checkpointing vs standard torch.save for FSDP/DDP
training on EKS.  NVRx LocalCheckpointManager writes checkpoints to
node-local storage (emptyDir backed by SSD/tmpfs) with tensor-aware
serialization, avoiding shared-filesystem bottlenecks.

Two checkpoint modes:
  --use_local_checkpoint    NVRx LocalCheckpointManager (node-local, tensor-aware)
  (default)                 Standard torch.save to the same local path

Supports both FSDP and DDP via --parallel_strategy flag.

Launched via torchrun:
    python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 ...
"""

import os
import sys
import time
import json
import argparse
import logging
import datetime

os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

try:
    from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import (
        LocalCheckpointManager,
    )
    from nvidia_resiliency_ext.checkpointing.local.basic_state_dict import (
        BasicTensorAwareStateDict,
    )

    NVRX_LOCAL_AVAILABLE = True
except ImportError:
    NVRX_LOCAL_AVAILABLE = False
    LocalCheckpointManager = None
    BasicTensorAwareStateDict = None

from distributed_utils import (
    add_distributed_args,
    create_model,
    wrap_model,
    create_dataloader,
    train_step,
    save_checkpoint,
    load_checkpoint,
)


def _move_tensors_to_cuda(obj):
    """Recursively move all tensors in a nested dict/list to CUDA.

    BasicTensorAwareStateDict requires every tensor to be on CUDA.
    Optimizer state dicts may contain CPU tensors (e.g. step counts),
    so we move them before wrapping.

    ShardedTensor objects (from FSDP local/sharded state dict) are
    left as-is since their local shards are already on CUDA.
    """
    # Skip ShardedTensor -- its local shards are already on CUDA
    if type(obj).__name__ == "ShardedTensor":
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.cuda() if not obj.is_cuda else obj
    elif isinstance(obj, dict):
        return {k: _move_tensors_to_cuda(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_move_tensors_to_cuda(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_move_tensors_to_cuda(v) for v in obj)
    return obj


def setup_logging(rank):
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="NVRx Local Checkpointing Test")

    # Model / training
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=1000)

    # Dataset
    parser.add_argument("--dataset_name", type=str, default="allenai/c4")
    parser.add_argument("--streaming", type=bool, default=True)
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

    # Local checkpointing toggle
    parser.add_argument(
        "--use_local_checkpoint",
        action="store_true",
        default=False,
        help="Use NVRx LocalCheckpointManager instead of standard torch.save",
    )

    # Training duration
    parser.add_argument("--training_duration_minutes", type=int, default=10)

    # NCCL timeout
    parser.add_argument(
        "--nccl_timeout_seconds",
        type=int,
        default=60,
        help="NCCL operation timeout (seconds).",
    )

    # Distributed strategy & dtype (from shared utils)
    add_distributed_args(parser)

    return parser.parse_args()


# ============================================================================
# Checkpoint: NVRx LocalCheckpointManager
# ============================================================================
def save_checkpoint_local(model, optimizer, step, ckpt_manager, rank, strategy):
    """Save using NVRx local checkpointing with tensor-aware serialization."""
    start_time = time.time()
    # state_dict() is collective under FSDP -- all ranks must call.
    # Under DDP it is non-collective but safe to call on all ranks.
    model_state = model.state_dict()
    optim_state = optimizer.state_dict()

    checkpoint = {
        "step": step,
        "model_state_dict": model_state,
        "optimizer_state_dict": optim_state,
    }

    # BasicTensorAwareStateDict requires ALL tensors on CUDA.
    # Optimizer state may contain CPU tensors (e.g. step counts).
    checkpoint = _move_tensors_to_cuda(checkpoint)

    ta_state_dict = BasicTensorAwareStateDict(checkpoint)
    ckpt_manager.save(ta_state_dict, iteration=step)
    save_time = time.time() - start_time

    return save_time


def load_checkpoint_local(ckpt_manager, model, optimizer, rank, strategy):
    """Load from NVRx local checkpoint if one exists."""
    logger = logging.getLogger(__name__)

    iteration = ckpt_manager.find_latest()
    if iteration < 0:
        logger.info("No local checkpoint found")
        return 0

    try:
        ta_state_dict, ckpt_part_id = ckpt_manager.load()
        ckpt = ta_state_dict.state_dict
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_step = ckpt["step"]
        logger.info(
            f"Loaded local checkpoint from step {start_step} (part_id={ckpt_part_id})"
        )
        return start_step
    except Exception as e:
        logger.warning(f"Failed to load local checkpoint: {e}")
    return 0


# ============================================================================
# Main training
# ============================================================================
def main():
    args = parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    logger = setup_logging(rank)

    # -----------------------------------------------------------------------
    # Initialize distributed
    # -----------------------------------------------------------------------
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=args.nccl_timeout_seconds),
        )

    use_local = args.use_local_checkpoint and NVRX_LOCAL_AVAILABLE
    mode_str = "LOCAL (NVRx)" if use_local else "STANDARD (torch.save)"

    logger.info("=" * 80)
    logger.info(f"Local Checkpointing Test - {mode_str}")
    logger.info("=" * 80)
    logger.info(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    logger.info(f"Parallel strategy: {args.parallel_strategy}")
    logger.info(f"Checkpoint mode: {mode_str}")
    logger.info(f"Checkpoint interval: every {args.checkpoint_interval} steps")
    logger.info(f"Checkpoint path: {args.checkpoint_path}")
    if args.use_local_checkpoint and not NVRX_LOCAL_AVAILABLE:
        logger.warning(
            "NVRx local checkpointing requested but not available! "
            "Falling back to standard torch.save."
        )

    # -----------------------------------------------------------------------
    # Initialize NVRx LocalCheckpointManager (if using local mode)
    # -----------------------------------------------------------------------
    ckpt_manager = None
    if use_local:
        local_ckpt_dir = os.path.join(args.checkpoint_path, "local_ckpt")
        logger.info(f"Initializing LocalCheckpointManager at {local_ckpt_dir}")
        ckpt_manager = LocalCheckpointManager(local_ckpt_dir)
        logger.info("LocalCheckpointManager initialized")

    # -----------------------------------------------------------------------
    # Create model and wrap with FSDP/DDP
    # -----------------------------------------------------------------------
    model, tokenizer = create_model(args.model_name, args.torch_dtype)
    model = wrap_model(model, args.parallel_strategy, local_rank, args.model_name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # -----------------------------------------------------------------------
    # Load checkpoint (if resuming)
    # -----------------------------------------------------------------------
    if use_local:
        start_step = load_checkpoint_local(
            ckpt_manager, model, optimizer, rank, args.parallel_strategy
        )
    else:
        start_step, _ = load_checkpoint(
            args.checkpoint_path, model, optimizer, rank, args.parallel_strategy
        )

    # -----------------------------------------------------------------------
    # Create dataloader
    # -----------------------------------------------------------------------
    dataloader = create_dataloader(args, tokenizer, rank, world_size)
    dataloader_iter = iter(dataloader)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    model.train()
    training_start_time = time.time()
    max_training_time = args.training_duration_minutes * 60

    step = start_step
    checkpoint_count = 0
    total_checkpoint_time = 0.0
    checkpoint_times = []
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

        # Log progress
        if step % 10 == 0:
            throughput = step / wall_elapsed if wall_elapsed > 0 else 0
            logger.info(
                f"Step {step}, Loss: {loss:.4f}, "
                f"Wall: {wall_elapsed:.0f}s, "
                f"Throughput: {throughput:.2f} steps/sec"
            )

        # Checkpoint saving
        if step % args.checkpoint_interval == 0:
            logger.info(f"Saving checkpoint at step {step}...")

            if use_local:
                ckpt_time = save_checkpoint_local(
                    model, optimizer, step, ckpt_manager, rank, args.parallel_strategy
                )
                logger.info(f"Local checkpoint saved in {ckpt_time:.3f}s")
            else:
                _, ckpt_time = save_checkpoint(
                    model,
                    optimizer,
                    step,
                    args.checkpoint_path,
                    rank,
                    args.parallel_strategy,
                )
                logger.info(f"Standard checkpoint saved in {ckpt_time:.3f}s")

            total_checkpoint_time += ckpt_time
            checkpoint_count += 1
            checkpoint_times.append(ckpt_time)

    # -----------------------------------------------------------------------
    # Training complete -- summary
    # -----------------------------------------------------------------------
    total_wall_time = time.time() - training_start_time
    active_training_time = total_wall_time - total_checkpoint_time

    if dist.is_initialized():
        dist.destroy_process_group()

    if rank == 0:
        throughput = step / total_wall_time if total_wall_time > 0 else 0
        ckpt_overhead = (
            (total_checkpoint_time / total_wall_time) * 100
            if total_wall_time > 0
            else 0
        )
        training_efficiency = (
            (active_training_time / total_wall_time) * 100
            if total_wall_time > 0
            else 100
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
        logger.info(f"Checkpoint Mode: {mode_str}")
        logger.info("")
        logger.info("TRAINING METRICS:")
        logger.info(f"  Steps: {step}/{args.max_steps} ({reason_label})")
        logger.info(f"  Termination: {termination_reason}")
        logger.info(f"  Total wall time: {total_wall_time:.1f}s")
        logger.info(f"  Active training time: {active_training_time:.1f}s")
        logger.info(f"  Effective throughput: {throughput:.2f} steps/sec")
        logger.info("")
        logger.info("CHECKPOINT METRICS:")
        logger.info(f"  Total checkpoints: {checkpoint_count}")
        logger.info(f"  Checkpoint interval: every {args.checkpoint_interval} steps")
        if checkpoint_count > 0:
            avg_ckpt = total_checkpoint_time / checkpoint_count
            min_ckpt = min(checkpoint_times)
            max_ckpt = max(checkpoint_times)
            logger.info(f"  Total checkpoint time: {total_checkpoint_time:.3f}s")
            logger.info(f"  Avg checkpoint time: {avg_ckpt:.3f}s")
            logger.info(f"  Min checkpoint time: {min_ckpt:.3f}s")
            logger.info(f"  Max checkpoint time: {max_ckpt:.3f}s")
            logger.info(f"  Checkpoint overhead: {ckpt_overhead:.2f}%")
            logger.info(f"  Training efficiency: {training_efficiency:.2f}%")
        logger.info("=" * 80)

        # Write results JSON for automated comparison
        results = {
            "experiment_type": "local_checkpointing",
            "mode": mode_str,
            "model": args.model_name,
            "world_size": world_size,
            "parallel_strategy": args.parallel_strategy,
            "max_steps": args.max_steps,
            "steps_completed": step,
            "termination_reason": termination_reason,
            "total_wall_time_seconds": round(total_wall_time, 1),
            "active_training_time_seconds": round(active_training_time, 1),
            "effective_throughput_steps_per_sec": round(throughput, 4),
            "checkpoint_count": checkpoint_count,
            "checkpoint_interval": args.checkpoint_interval,
            "total_checkpoint_time_seconds": round(total_checkpoint_time, 3),
            "avg_checkpoint_time_seconds": round(
                total_checkpoint_time / checkpoint_count, 3
            )
            if checkpoint_count > 0
            else None,
            "min_checkpoint_time_seconds": round(min(checkpoint_times), 3)
            if checkpoint_times
            else None,
            "max_checkpoint_time_seconds": round(max(checkpoint_times), 3)
            if checkpoint_times
            else None,
            "checkpoint_overhead_pct": round(ckpt_overhead, 2),
            "training_efficiency_pct": round(training_efficiency, 2),
        }
        try:
            results_path = os.path.join(args.checkpoint_path, "results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {results_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


if __name__ == "__main__":
    main()
