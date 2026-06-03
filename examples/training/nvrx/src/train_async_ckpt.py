#!/usr/bin/env python3
"""
NVRx Async Checkpointing Training Script

Compares sync vs async checkpoint performance with FSDP/DDP training.
Supports both FSDP and DDP via --parallel_strategy flag.
"""

import gc
import os
import sys
import time
import argparse
import logging

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

# Import NVRx async checkpointing
try:
    from nvidia_resiliency_ext.checkpointing.async_ckpt.torch_ckpt import (
        TorchAsyncCheckpoint,
    )

    NVRX_ASYNC_AVAILABLE = True
except ImportError:
    NVRX_ASYNC_AVAILABLE = False
    print("WARNING: NVRx async checkpointing not available")

from distributed_utils import (
    add_distributed_args,
    create_model,
    wrap_model,
    create_dataloader,
    train_step,
    save_checkpoint as save_checkpoint_sync,
    setup_logging,
)
from metrics_collector import MetricsCollector


def str2bool(v):
    """Argparse type for boolean arguments."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def _to_cpu(obj):
    """Recursively move all tensors in a nested dict/list to CPU.

    This is critical for async checkpointing: state_dict() tensors live on GPU,
    and the async writer holds a reference until I/O completes. Without CPU
    offload, GPU memory accumulates across checkpoint intervals and causes OOM.

    Also handles ``ShardedTensor`` objects produced by FSDP
    ``LOCAL_STATE_DICT``: extracts the local shard and moves it to CPU.
    """
    # Handle ShardedTensor from FSDP LOCAL_STATE_DICT
    if type(obj).__name__ == "ShardedTensor":
        # Extract the local shard(s) and move to CPU.
        # Each rank has exactly one local shard for FSDP FULL_SHARD.
        local_shards = obj.local_shards()
        if local_shards:
            return local_shards[0].tensor.cpu()
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def parse_args():
    parser = argparse.ArgumentParser(
        description="NVRx FSDP/DDP Training with Async Checkpointing"
    )

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

    # Checkpointing args
    parser.add_argument("--checkpoint_enabled", type=str2bool, default=True)
    parser.add_argument("--checkpoint_path", type=str, default="/checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    parser.add_argument("--use_async_checkpoint", type=str2bool, default=False)

    # Training duration
    parser.add_argument("--training_duration_minutes", type=int, default=15)

    # Distributed strategy & dtype (from shared utils)
    add_distributed_args(parser)

    return parser.parse_args()


def main():
    args = parse_args()

    # Read distributed env vars (set by torchrun)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    logger = setup_logging(rank, args.log_level, args.log_all_ranks)

    logger.info("=" * 80)
    logger.info("NVRx Async Checkpointing Training Starting")
    logger.info("=" * 80)
    logger.info(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    logger.info(f"Parallel Strategy: {args.parallel_strategy}")
    logger.info(f"Async Checkpointing: {args.use_async_checkpoint}")
    logger.info(f"NVRx Available: {NVRX_ASYNC_AVAILABLE}")

    # Initialize async checkpointing if enabled
    async_ckpt = None
    if args.use_async_checkpoint and NVRX_ASYNC_AVAILABLE:
        logger.info("Initializing NVRx async checkpointing...")
        async_ckpt = TorchAsyncCheckpoint(persistent_queue=True)
    elif args.use_async_checkpoint and not NVRX_ASYNC_AVAILABLE:
        logger.warning("Async checkpointing requested but NVRx not available!")
        logger.warning("Falling back to synchronous checkpointing")
        args.use_async_checkpoint = False

    # Initialize metrics
    metrics = MetricsCollector(rank, world_size, output_dir=args.checkpoint_path)
    metrics.start_training()

    try:
        # Create model and tokenizer (shared utility -- uses args.torch_dtype)
        model, tokenizer = create_model(args.model_name, args.torch_dtype)

        # Wrap model with FSDP or DDP
        model = wrap_model(model, args.parallel_strategy, local_rank, args.model_name)

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        # Create dataloader
        dataloader = create_dataloader(args, tokenizer, rank, world_size)
        dataloader_iter = iter(dataloader)

        # Training loop
        model.train()
        step = 0
        training_start_time = time.time()
        max_training_time = args.training_duration_minutes * 60

        # Track checkpoint metrics
        total_checkpoint_time = 0
        checkpoint_count = 0
        checkpoint_times = []
        checkpoint_sizes = []
        termination_reason = "max_steps_reached"

        logger.info("Starting training loop...")

        while step < args.max_steps:
            # Check if training time exceeded
            elapsed = time.time() - training_start_time
            if elapsed >= max_training_time:
                logger.info(
                    f"Training time limit reached ({args.training_duration_minutes} minutes)"
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

            # Log progress every step and update metrics
            elapsed = time.time() - training_start_time
            metrics.log_step(step, loss, elapsed)
            if step % 10 == 0:
                logger.info(f"Step {step}, Loss: {loss:.4f}, Time: {elapsed:.1f}s")

            # Checkpoint saving
            if args.checkpoint_enabled and step % args.checkpoint_interval == 0:
                logger.info(f"Saving checkpoint at step {step}...")

                if args.use_async_checkpoint and async_ckpt:
                    # Async checkpointing -- unique to this script
                    ckpt_start = time.time()

                    # Wait for previous async write before building new state_dict
                    async_ckpt.finalize_async_save(blocking=True, no_dist=False)

                    # Use LOCAL_STATE_DICT for FSDP to avoid the expensive
                    # FULL_STATE_DICT all-gather (~80GB/rank for LLaMA-8B).
                    # Each rank saves only its local shard (~2.8GB).
                    if args.parallel_strategy == "fsdp" and isinstance(model, FSDP):
                        FSDP.set_state_dict_type(model, StateDictType.LOCAL_STATE_DICT)

                    state_dict = model.state_dict()
                    checkpoint = {
                        "step": step,
                        "model_state_dict": state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "parallel_strategy": args.parallel_strategy,
                    }

                    # Move to CPU before enqueuing. Safe with LOCAL_STATE_DICT
                    # since tensors are local shards (no shared memory from
                    # all-gather that caused SIGBUS with FULL_STATE_DICT).
                    checkpoint = _to_cpu(checkpoint)

                    path = os.path.join(
                        args.checkpoint_path, f"checkpoint_step_{step}_rank_{rank}.pt"
                    )

                    async_ckpt.async_save(checkpoint, path)

                    del checkpoint, state_dict
                    gc.collect()
                    torch.cuda.empty_cache()

                    ckpt_time = time.time() - ckpt_start

                    logger.info(f"Async checkpoint scheduled in {ckpt_time:.3f}s")
                    total_checkpoint_time += ckpt_time
                    checkpoint_count += 1
                    checkpoint_times.append(ckpt_time)
                    metrics.log_checkpoint_save(ckpt_time)
                else:
                    # Synchronous checkpointing -- uses LOCAL_STATE_DICT
                    # to avoid the expensive FULL_STATE_DICT all-gather,
                    # same as the async path above.
                    ckpt_start = time.time()

                    if args.parallel_strategy == "fsdp" and isinstance(model, FSDP):
                        FSDP.set_state_dict_type(model, StateDictType.LOCAL_STATE_DICT)

                    checkpoint = {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "parallel_strategy": args.parallel_strategy,
                    }
                    checkpoint = _to_cpu(checkpoint)

                    path = os.path.join(
                        args.checkpoint_path, f"checkpoint_step_{step}_rank_{rank}.pt"
                    )
                    torch.save(checkpoint, path)
                    del checkpoint
                    gc.collect()
                    torch.cuda.empty_cache()

                    ckpt_time = time.time() - ckpt_start
                    logger.info(f"Sync checkpoint saved in {ckpt_time:.3f}s (blocking)")
                    total_checkpoint_time += ckpt_time
                    checkpoint_count += 1
                    checkpoint_times.append(ckpt_time)
                    metrics.log_checkpoint_save(ckpt_time)
                    # Measure checkpoint file size
                    if os.path.exists(path):
                        ckpt_size_mb = os.path.getsize(path) / (1024 * 1024)
                        checkpoint_sizes.append(ckpt_size_mb)
                        logger.info(f"Checkpoint size: {ckpt_size_mb:.1f} MB")

        # Finalize async checkpointing if used
        if async_ckpt:
            logger.info("Finalizing async checkpointing...")
            finalize_start = time.time()
            async_ckpt.finalize_async_save(blocking=True, no_dist=False)
            # close() may not exist in all NVRx versions
            if hasattr(async_ckpt, "close"):
                async_ckpt.close()
            finalize_time = time.time() - finalize_start
            logger.info(f"Async finalization completed in {finalize_time:.3f}s")

            # Measure async checkpoint sizes after finalization
            for ckpt_file in sorted(os.listdir(args.checkpoint_path)):
                if ckpt_file.endswith(f"_rank_{rank}.pt"):
                    ckpt_path = os.path.join(args.checkpoint_path, ckpt_file)
                    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
                    checkpoint_sizes.append(ckpt_size_mb)

        logger.info("Training completed successfully!")
        metrics.set_run_completion(args.max_steps, termination_reason)
        metrics.end_training(success=True)

        # Print checkpoint summary
        if checkpoint_count > 0:
            total_wall_time = time.time() - training_start_time
            avg_checkpoint_time = total_checkpoint_time / checkpoint_count
            checkpoint_overhead_pct = (total_checkpoint_time / total_wall_time) * 100
            training_efficiency_pct = 100 - checkpoint_overhead_pct
            effective_throughput = step / (total_wall_time - total_checkpoint_time)
            raw_throughput = step / total_wall_time
            avg_step_time = total_wall_time / step
            steps_lost_per_ckpt = avg_checkpoint_time / avg_step_time

            # Run completion status
            reason_label = (
                "completed"
                if termination_reason == "max_steps_reached"
                else "time-limited"
            )

            logger.info("=" * 80)
            logger.info("CHECKPOINT PERFORMANCE SUMMARY")
            logger.info("=" * 80)
            logger.info(
                f"Mode: {'Async (NVRx)' if args.use_async_checkpoint else 'Sync (torch.save)'}"
            )
            logger.info(f"Model: {args.model_name}")
            logger.info(f"World Size: {world_size}")
            logger.info(f"Parallel Strategy: {args.parallel_strategy}")
            logger.info("")
            logger.info("TRAINING METRICS:")
            logger.info(f"  Steps: {step}/{args.max_steps} ({reason_label})")
            logger.info(f"  Termination: {termination_reason}")
            logger.info(f"  Total wall time: {total_wall_time:.1f}s")
            logger.info(f"  Raw throughput: {raw_throughput:.2f} steps/sec")
            logger.info(
                f"  Effective throughput (excl. ckpt): {effective_throughput:.2f} steps/sec"
            )
            logger.info("")
            logger.info("CHECKPOINT METRICS:")
            logger.info(f"  Total checkpoints: {checkpoint_count}")
            logger.info(
                f"  Checkpoint interval: every {args.checkpoint_interval} steps"
            )
            logger.info(f"  Avg checkpoint time: {avg_checkpoint_time:.3f}s")
            logger.info(f"  Total checkpoint time: {total_checkpoint_time:.3f}s")
            if checkpoint_times:
                logger.info(f"  Min checkpoint time: {min(checkpoint_times):.3f}s")
                logger.info(
                    f"  Max checkpoint time (P100): {max(checkpoint_times):.3f}s"
                )
            if checkpoint_sizes:
                avg_size = sum(checkpoint_sizes) / len(checkpoint_sizes)
                logger.info(f"  Avg checkpoint size: {avg_size:.1f} MB")
                logger.info(
                    f"  Write bandwidth: {avg_size / avg_checkpoint_time:.1f} MB/s"
                )
            logger.info("")
            logger.info("EFFICIENCY METRICS:")
            logger.info(f"  Checkpoint overhead: {checkpoint_overhead_pct:.2f}%")
            logger.info(f"  Training efficiency: {training_efficiency_pct:.2f}%")
            logger.info(f"  Steps lost per checkpoint: {steps_lost_per_ckpt:.2f}")
            logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.warning("Training interrupted (SIGTERM/Ctrl-C)")
        metrics.end_training(success=False, error="interrupted")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        metrics.end_training(success=False, error=str(e))
        raise

    finally:
        # Finalize any pending async writes before shutdown
        if async_ckpt:
            try:
                async_ckpt.finalize_async_save(blocking=True, no_dist=False)
            except Exception:
                pass

        if world_size > 1:
            dist.destroy_process_group()

        # Print final metrics
        metrics.print_summary()


if __name__ == "__main__":
    main()
