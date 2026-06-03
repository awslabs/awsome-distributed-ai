"""
Main training script for QLoRA fine-tuning of Qwen3-8B.

This script orchestrates the complete training pipeline:
1. Load configuration
2. Setup model with QLoRA
3. Prepare dataset
4. Train with Hugging Face Trainer (supports DDP + DeepSpeed)
5. Save fine-tuned model

Supports distributed training via:
- DDP (Data Parallelism): splits batches across GPUs
- DeepSpeed ZeRO-2: shards optimizer states and gradients across GPUs
- DeepSpeed ZeRO-3: full parameter sharding for maximum memory savings

Usage (single GPU):
    python src/train.py --config_path configs/training_config_zero2.yaml

Usage (multi-GPU with torchrun):
    torchrun --nproc_per_node=4 src/train.py --config_path configs/training_config_zero2.yaml
"""

import os
import sys
import glob
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from src.config import FullConfig
from src.model_setup import setup_model_and_tokenizer, get_gpu_memory_info, log_gpu_memory
from src.data_preparation import (
    load_medical_dataset,
    preprocess_dataset,
    get_data_collator,
    inspect_sample
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-8B with QLoRA on medical reasoning data"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/training_config_zero2.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (or 'auto' to find latest)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Override max samples for quick testing"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load model and data but don't train (for testing)"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set by torchrun)"
    )
    return parser.parse_args()


def check_environment():
    """Verify the environment is properly configured."""
    logger.info("=" * 60)
    logger.info("Environment Check")
    logger.info("=" * 60)

    # Check Python version
    logger.info(f"Python version: {sys.version}")

    # Check PyTorch
    logger.info(f"PyTorch version: {torch.__version__}")

    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA available: Yes")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    else:
        logger.error("CUDA not available! Training requires a GPU.")
        sys.exit(1)

    # Check transformers version (Qwen3 requires >= 4.51.0)
    import transformers
    logger.info(f"Transformers version: {transformers.__version__}")

    version_parts = transformers.__version__.split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])
    if major < 4 or (major == 4 and minor < 51):
        logger.warning("Qwen3 requires transformers >= 4.51.0. Please upgrade.")

    # Check bitsandbytes
    import bitsandbytes
    logger.info(f"BitsAndBytes version: {bitsandbytes.__version__}")

    # Check PEFT
    import peft
    logger.info(f"PEFT version: {peft.__version__}")

    # Check DeepSpeed availability
    try:
        import deepspeed
        logger.info(f"DeepSpeed version: {deepspeed.__version__}")
    except ImportError:
        logger.info("DeepSpeed: not installed (DDP-only mode available)")

    # Check distributed setup
    if torch.cuda.device_count() > 1:
        logger.info(f"Multi-GPU detected: {torch.cuda.device_count()} GPUs available")
        if "RANK" in os.environ:
            logger.info(f"  Distributed rank: {os.environ.get('RANK')}")
            logger.info(f"  Local rank: {os.environ.get('LOCAL_RANK')}")
            logger.info(f"  World size: {os.environ.get('WORLD_SIZE')}")
    else:
        logger.info("Single-GPU mode")

    logger.info("=" * 60)


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint directory in the output directory.

    Args:
        output_dir: Directory containing checkpoint-* subdirectories

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None

    # Sort by step number (checkpoint-100, checkpoint-200, etc.)
    # Filter out non-numeric checkpoint names (e.g., checkpoint-emergency)
    checkpoints = [c for c in checkpoints if c.split("-")[-1].isdigit()]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    latest = checkpoints[-1]
    logger.info(f"Found {len(checkpoints)} checkpoint(s), latest: {latest}")
    return latest


def _is_deepspeed_checkpoint(checkpoint_path: str) -> bool:
    """
    Check if a checkpoint directory contains DeepSpeed state files.

    DeepSpeed checkpoints contain global_step*/mp_rank_* directories,
    while vanilla HuggingFace checkpoints only contain adapter weights
    and trainer_state.json.

    Args:
        checkpoint_path: Path to a checkpoint-* directory

    Returns:
        True if the checkpoint was saved with DeepSpeed
    """
    # DeepSpeed saves state in global_step* subdirectories
    ds_patterns = [
        os.path.join(checkpoint_path, "global_step*"),
        os.path.join(checkpoint_path, "zero_pp_rank_*"),
    ]
    for pattern in ds_patterns:
        if glob.glob(pattern):
            return True
    return False


def resolve_checkpoint(args, config) -> Optional[str]:
    """
    Resolve the checkpoint path for resuming training.

    Supports:
    - Explicit path: --resume_from_checkpoint=/path/to/checkpoint-1500
    - Auto-detect: --resume_from_checkpoint=auto (finds latest in output_dir)
    - No flag: automatically checks output_dir for existing checkpoints

    Also validates checkpoint format compatibility: if DeepSpeed is enabled
    but the checkpoint was saved without DeepSpeed, it skips the incompatible
    checkpoint and starts fresh (DeepSpeed cannot load vanilla HF checkpoints).

    Args:
        args: Parsed command-line arguments
        config: Full configuration

    Returns:
        Checkpoint path or None
    """
    using_deepspeed = config.parallel.strategy.startswith("deepspeed")

    def _validate_checkpoint(checkpoint_path: Optional[str]) -> Optional[str]:
        """Validate checkpoint compatibility with current parallel strategy."""
        if checkpoint_path is None:
            return None

        if using_deepspeed and not _is_deepspeed_checkpoint(checkpoint_path):
            logger.warning(
                f"Checkpoint {checkpoint_path} was saved without DeepSpeed, "
                f"but current strategy is '{config.parallel.strategy}'. "
                f"DeepSpeed cannot resume from a non-DeepSpeed checkpoint. "
                f"Starting training from scratch. Future checkpoints will be "
                f"DeepSpeed-compatible and auto-resume will work."
            )
            return None
        return checkpoint_path

    if args.resume_from_checkpoint == "auto":
        checkpoint = find_latest_checkpoint(config.training.output_dir)
        if checkpoint:
            checkpoint = _validate_checkpoint(checkpoint)
            if checkpoint:
                logger.info(f"Auto-resume: will resume from {checkpoint}")
            else:
                logger.info("Auto-resume: starting from scratch (checkpoint incompatible)")
        else:
            logger.info("Auto-resume: no checkpoints found, starting from scratch")
        return checkpoint

    if args.resume_from_checkpoint is not None:
        if os.path.isdir(args.resume_from_checkpoint):
            checkpoint = _validate_checkpoint(args.resume_from_checkpoint)
            if checkpoint:
                logger.info(f"Will resume from specified checkpoint: {checkpoint}")
            return checkpoint
        else:
            logger.warning(f"Checkpoint path not found: {args.resume_from_checkpoint}")
            return None

    # Default: auto-detect if checkpoints exist
    checkpoint = find_latest_checkpoint(config.training.output_dir)
    if checkpoint:
        checkpoint = _validate_checkpoint(checkpoint)
        if checkpoint:
            logger.info(f"Found existing checkpoint, automatically resuming from: {checkpoint}")
    return checkpoint


def create_training_arguments(config: FullConfig) -> TrainingArguments:
    """
    Create Hugging Face TrainingArguments from config.

    Configures DDP and/or DeepSpeed based on parallel config settings.

    Args:
        config: Full configuration object

    Returns:
        TrainingArguments instance
    """
    strategy = config.parallel.strategy

    kwargs = dict(
        output_dir=config.training.output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=config.training.num_train_epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        optim=config.training.optim,
        max_grad_norm=config.training.max_grad_norm,
        gradient_checkpointing=config.parallel.gradient_checkpointing,
        # ZeRO-3 shards parameters across ranks, so during gradient checkpoint
        # recomputation non-owning ranks see shape-[0] tensors.  The strict
        # metadata check in non-reentrant checkpointing rejects this.
        # use_reentrant=True skips that check and is compatible with ZeRO-3.
        gradient_checkpointing_kwargs={
            "use_reentrant": strategy == "deepspeed_zero3"
        },
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        logging_steps=config.training.logging_steps,
        logging_dir=f"{config.training.output_dir}/logs",
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        eval_strategy="steps",
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        group_by_length=config.training.group_by_length,
        report_to=config.training.report_to,
        seed=config.training.seed,
        dataloader_num_workers=config.data.num_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    # Configure DeepSpeed if requested
    if strategy.startswith("deepspeed"):
        ds_config_path = config.parallel.deepspeed_config_path
        if ds_config_path is None:
            # Use default config based on strategy
            if strategy == "deepspeed_zero2":
                ds_config_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "configs", "deepspeed_zero2.json"
                )
            elif strategy == "deepspeed_zero3":
                ds_config_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "configs", "deepspeed_zero3.json"
                )

        if ds_config_path and os.path.exists(ds_config_path):
            kwargs["deepspeed"] = ds_config_path
            logger.info(f"DeepSpeed enabled with config: {ds_config_path}")
        else:
            logger.warning(
                f"DeepSpeed config not found at {ds_config_path}, "
                f"falling back to DDP-only"
            )

    return TrainingArguments(**kwargs)


def main():
    """Main training function."""
    args = parse_args()

    # Configure logging: all ranks log to stdout, only rank 0 logs to file.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    handlers = [logging.StreamHandler(sys.stdout)]
    if local_rank in (-1, 0):
        output_dir = args.output_dir or "."
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(output_dir, "training.log")))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )

    logger.info("=" * 60)
    logger.info("QLoRA Fine-tuning: Qwen3-8B on Medical Reasoning")
    logger.info("=" * 60)

    # Check environment
    check_environment()

    # Load configuration
    logger.info(f"Loading config from: {args.config_path}")
    if os.path.exists(args.config_path):
        config = FullConfig.from_yaml(args.config_path)
        logger.info("Configuration loaded from YAML file")
    else:
        config = FullConfig()
        logger.info("Using default configuration")

    # Override settings from command line
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.max_samples is not None:
        config.data.max_samples = args.max_samples

    # Create output directory
    Path(config.training.output_dir).mkdir(parents=True, exist_ok=True)

    # Save the configuration used
    config.to_yaml(f"{config.training.output_dir}/config.yaml")

    # Resolve checkpoint for resume
    resume_checkpoint = resolve_checkpoint(args, config)

    # Log parallelism strategy
    num_gpus = torch.cuda.device_count()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    strategy = config.parallel.strategy
    logger.info(f"Parallelism strategy: {strategy}")
    logger.info(f"Available GPUs: {num_gpus}")
    if num_gpus > 1:
        eff_batch = (
            config.training.per_device_train_batch_size
            * config.training.gradient_accumulation_steps
            * num_gpus
        )
        logger.info(f"Effective batch size: {eff_batch} "
                     f"({config.training.per_device_train_batch_size} per-GPU × "
                     f"{config.training.gradient_accumulation_steps} accum × "
                     f"{num_gpus} GPUs)")
    else:
        eff_batch = (
            config.training.per_device_train_batch_size
            * config.training.gradient_accumulation_steps
        )
        logger.info(f"Effective batch size: {eff_batch}")

    # Pin each process to its assigned GPU before loading the model.
    # BitsAndBytes uses torch.cuda.current_device() internally for
    # quantized weight allocation.
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)

    # Setup model and tokenizer
    logger.info("\n" + "=" * 60)
    logger.info("Setting up model...")
    logger.info("=" * 60)
    log_gpu_memory("Before model loading")
    model, tokenizer = setup_model_and_tokenizer(config)

    # Log GPU memory after model loading
    get_gpu_memory_info()
    log_gpu_memory("After model loading")

    # Load and prepare dataset
    logger.info("\n" + "=" * 60)
    logger.info("Preparing dataset...")
    logger.info("=" * 60)

    train_dataset, eval_dataset = load_medical_dataset(
        dataset_name=config.data.dataset_name,
        config_name=config.data.dataset_config,
        max_samples=config.data.max_samples,
        train_split=config.data.train_split,
    )

    # Preprocess datasets
    train_dataset = preprocess_dataset(
        train_dataset,
        tokenizer,
        config.model.max_seq_length
    )
    eval_dataset = preprocess_dataset(
        eval_dataset,
        tokenizer,
        config.model.max_seq_length
    )

    # Inspect a sample
    logger.info("\nSample from training data:")
    inspect_sample(train_dataset, tokenizer, idx=0)

    # Dry run check
    if args.dry_run:
        logger.info("Dry run complete. Exiting without training.")
        return

    # Create training arguments (handles DeepSpeed config)
    training_args = create_training_arguments(config)

    # Create data collator
    data_collator = get_data_collator(tokenizer)

    # Create trainer
    logger.info("\n" + "=" * 60)
    logger.info("Creating Trainer...")
    logger.info("=" * 60)

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3)
            ]
        )
    except Exception as e:
        logger.error(f"Failed to create Trainer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Log training info
    total_steps = (
        len(train_dataset) //
        (training_args.per_device_train_batch_size
         * training_args.gradient_accumulation_steps
         * max(1, world_size))
        * training_args.num_train_epochs
    )
    logger.info(f"Total training steps (approx): {total_steps}")
    logger.info(f"Resume from checkpoint: {resume_checkpoint or 'None (starting fresh)'}")

    # Start training with error handling
    logger.info("\n" + "=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    log_gpu_memory("Before training start")

    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "cuda error" in error_msg:
            logger.error(f"GPU Error during training: {e}")
            log_gpu_memory("At crash point")

            # Clear GPU cache
            torch.cuda.empty_cache()
            log_gpu_memory("After cache clear")

            # Attempt to save current state before exiting
            logger.info("Attempting emergency checkpoint save...")
            try:
                emergency_path = f"{config.training.output_dir}/checkpoint-emergency"
                trainer.save_model(emergency_path)
                tokenizer.save_pretrained(emergency_path)
                logger.info(f"Emergency checkpoint saved to: {emergency_path}")
            except Exception as save_err:
                logger.error(f"Failed to save emergency checkpoint: {save_err}")

            logger.error(
                "Training failed due to GPU memory pressure. Suggestions:\n"
                "  1. Reduce per_device_train_batch_size to 1\n"
                "  2. Reduce max_seq_length (e.g., 1536 or 1024)\n"
                "  3. Use DeepSpeed ZeRO-2/3 with more GPUs\n"
                "  4. Use a GPU with more VRAM\n"
                "  5. Resume from the latest checkpoint after fixing config"
            )
            sys.exit(1)
        else:
            raise

    # Save final model
    logger.info("\n" + "=" * 60)
    logger.info("Saving model...")
    logger.info("=" * 60)
    log_gpu_memory("Before model save")

    # Save the full model (with merged weights if desired)
    final_path = f"{config.training.output_dir}/final_model"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Full model saved to: {final_path}")

    # Save just the LoRA adapter (much smaller)
    # Use trainer.save_model to handle ZeRO-3 parameter gathering correctly
    lora_path = f"{config.training.output_dir}/lora_adapter"
    trainer.save_model(lora_path)
    logger.info(f"LoRA adapter saved to: {lora_path}")

    # Log final metrics
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

    # Print final evaluation
    eval_results = trainer.evaluate()
    logger.info(f"Final eval loss: {eval_results['eval_loss']:.4f}")

    logger.info(f"\nModel artifacts saved to: {config.training.output_dir}")
    logger.info("To use the fine-tuned model, load the LoRA adapter with PEFT")
    log_gpu_memory("After training complete")


if __name__ == "__main__":
    main()
