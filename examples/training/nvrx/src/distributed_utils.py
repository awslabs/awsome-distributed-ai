"""
Shared distributed training utilities.

Provides strategy-agnostic model creation, wrapping, checkpointing, and
dataloader helpers that work with both FSDP and DDP.

Usage in training scripts::

    from distributed_utils import (
        add_distributed_args,
        create_model,
        wrap_model,
        create_dataloader,
        train_step,
        save_checkpoint,
        load_checkpoint,
    )
"""

import os
import sys
import logging
import functools
import gc

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported parallelism strategies
# ---------------------------------------------------------------------------
STRATEGIES = ("fsdp", "ddp")

# Map user-facing dtype strings to torch dtypes
_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _to_cpu(obj):
    """Recursively move all tensors in a nested dict/list to CPU.

    Also handles ``ShardedTensor`` objects produced by FSDP
    ``LOCAL_STATE_DICT``: extracts the local shard and moves it to CPU.
    """
    if type(obj).__name__ == "ShardedTensor":
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


def _find_latest_dcp_checkpoint(checkpoint_path: str) -> int:
    """Find the latest DCP checkpoint step number in *checkpoint_path*.

    DCP checkpoints are stored in subdirectories named ``dcp_step_<N>/``.
    Returns the highest step number found, or -1 if no DCP checkpoints exist.
    """
    import glob as glob_module

    pattern = os.path.join(checkpoint_path, "dcp_step_*")
    dirs = [d for d in glob_module.glob(pattern) if os.path.isdir(d)]
    if not dirs:
        return -1

    steps = []
    for d in dirs:
        basename = os.path.basename(d)
        try:
            step = int(basename.split("_")[-1])
            steps.append(step)
        except (ValueError, IndexError):
            continue

    return max(steps) if steps else -1


# ---------------------------------------------------------------------------
# CLI argument helpers
# ---------------------------------------------------------------------------
def add_distributed_args(parser):
    """Add --parallel_strategy, --torch_dtype, --model_name to *parser*.

    These args are common across every training script.  Call this from each
    script's ``parse_args()`` so the flags are consistent everywhere.
    """
    parser.add_argument(
        "--parallel_strategy",
        type=str,
        default="fsdp",
        choices=STRATEGIES,
        help="Distributed parallelism strategy: fsdp or ddp (default: fsdp)",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=list(_DTYPE_MAP.keys()),
        help="Model precision: bfloat16, float16, float32 (default: bfloat16)",
    )
    # --model_name is already defined in most scripts, so only add if missing
    if not any(
        a.option_strings and "--model_name" in a.option_strings for a in parser._actions
    ):
        parser.add_argument(
            "--model_name",
            type=str,
            default="gpt2",
            help="HuggingFace model name or path",
        )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for rank 0 (default: INFO)",
    )
    parser.add_argument(
        "--log_all_ranks",
        action="store_true",
        default=False,
        help="Log at full verbosity on all ranks (default: rank 0 only)",
    )
    return parser


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "datasets",
    "urllib3",
    "filelock",
)


def setup_logging(rank: int, log_level: str = "INFO", log_all_ranks: bool = False):
    """Configure logging so only rank 0 is verbose by default.

    - Rank 0: logs at *log_level* (default INFO).
    - Rank != 0: logs at WARNING only (errors/warnings still surface).
    - ``log_all_ranks=True`` restores full output on every rank.
    - Third-party noisy loggers are silenced to WARNING on all ranks.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    effective_level = level if (rank == 0 or log_all_ranks) else logging.WARNING

    logging.basicConfig(
        level=effective_level,
        format=f"[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert a string dtype name to a ``torch.dtype``."""
    return _DTYPE_MAP[dtype_str]


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------
def create_model(model_name: str, dtype_str: str = "bfloat16"):
    """Load a HuggingFace causal LM and its tokenizer.

    The model is moved to the current CUDA device and cast to *dtype_str*.
    The tokenizer's pad token is set to eos_token if not already defined.

    Returns:
        (model, tokenizer) tuple
    """
    torch_dtype = get_torch_dtype(dtype_str)

    logger.info(f"Loading model: {model_name} (dtype={dtype_str})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    model = model.to(torch.cuda.current_device())
    return model, tokenizer


# ---------------------------------------------------------------------------
# Model wrapping (FSDP / DDP)
# ---------------------------------------------------------------------------
def wrap_model(model, strategy: str, local_rank: int, model_name: str = "gpt2"):
    """Wrap *model* with the requested distributed parallelism strategy.

    Args:
        model: A PyTorch module already on the correct CUDA device.
        strategy: ``"fsdp"`` or ``"ddp"``.
        local_rank: Local rank (used as device id).
        model_name: Model name, used to select FSDP auto-wrap policy.

    Returns:
        The wrapped model.
    """
    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from fsdp_config import get_fsdp_config

        fsdp_config = get_fsdp_config(model_name)
        model = FSDP(model, device_id=torch.cuda.current_device(), **fsdp_config)
        logger.info("Model wrapped with FSDP")

    elif strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model, device_ids=[local_rank])
        logger.info("Model wrapped with DDP")

    else:
        raise ValueError(
            f"Unknown parallel strategy: {strategy!r}. Choose from {STRATEGIES}"
        )

    return model


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------
def create_dataloader(args, tokenizer, rank: int, world_size: int):
    """Create a DataLoader for causal-LM training.

    Supports two modes:
      1. **Local dataset** (``args.dataset_path`` is set) -- loads a
         pre-downloaded Arrow dataset from disk with zero HF API calls.
      2. **Streaming** (default) -- streams from HuggingFace Hub.

    Expects *args* to have ``dataset_name``, ``streaming``, ``max_seq_length``,
    and ``batch_size`` attributes.  ``dataset_path`` is optional.
    """
    dataset_path = getattr(args, "dataset_path", None)
    if dataset_path:
        logger.info("Loading pre-downloaded dataset from %s", dataset_path)
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset(
            args.dataset_name,
            "en",
            split="train",
            streaming=args.streaming,
            trust_remote_code=True,
        )

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
        dataset = dataset.shard(num_shards=world_size, index=rank)

    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------
def train_step(model, batch, optimizer):
    """Execute one training step.  Returns the scalar loss value."""
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()
    labels = batch["labels"].cuda()

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------
def save_checkpoint(
    model,
    optimizer,
    step: int,
    checkpoint_path: str,
    rank: int,
    strategy: str,
    call_wrapper=None,
    extra_state: dict | None = None,
):
    """Save a training checkpoint using DCP for FSDP or torch.save for DDP.

    For FSDP, uses ``torch.distributed.checkpoint`` (DCP) which natively
    handles sharded state dicts.  DCP writes per-rank shard files into a
    subdirectory ``dcp_step_<N>/`` and can load them back correctly
    regardless of the world size or sharding.  Extra state (step number,
    ft_client state, etc.) is saved as a separate ``extra_step_<N>.pt``
    file alongside the DCP directory.

    For DDP, all ranks save their full model state via ``torch.save``.

    Args:
        model: Wrapped model (FSDP or DDP).
        optimizer: Optimizer.
        step: Current training step.
        checkpoint_path: Directory to write into.
        rank: Global rank.
        strategy: ``"fsdp"`` or ``"ddp"``.
        call_wrapper: Optional NVRx ``CallWrapper`` for atomic saves.
        extra_state: Optional dict of additional state to persist
            (e.g. ``ft_client.state_dict()``).

    Returns:
        (path, save_time) tuple.
    """
    os.makedirs(checkpoint_path, exist_ok=True)

    import time

    start_time = time.time()

    if strategy == "fsdp":
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            get_state_dict,
            StateDictOptions,
        )

        # Get FSDP-aware state dicts using the new DCP API.
        # This returns shardable state dicts that DCP can save natively.
        model_state, optim_state = get_state_dict(
            model,
            optimizer,
            options=StateDictOptions(full_state_dict=False, cpu_offload=True),
        )

        state_dict = {
            "model": model_state,
            "optimizer": optim_state,
        }

        # DCP save is collective -- all ranks must call.
        ckpt_dir = os.path.join(checkpoint_path, f"dcp_step_{step}")
        if call_wrapper is not None:
            with call_wrapper.atomic():
                dcp.save(state_dict, checkpoint_id=ckpt_dir)
        else:
            dcp.save(state_dict, checkpoint_id=ckpt_dir)

        # Save extra state (step number, ft_client, etc.) as a plain file.
        # Only rank 0 writes this to avoid conflicts.
        extra_path = os.path.join(checkpoint_path, f"extra_step_{step}.pt")
        extra = {"step": step, "parallel_strategy": strategy}
        if extra_state:
            extra.update(extra_state)
        if rank == 0:
            torch.save(extra, extra_path)

        path = ckpt_dir

    else:
        # DDP: save full state dict via torch.save (non-collective)
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

        checkpoint = {
            "step": step,
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "parallel_strategy": strategy,
        }
        if extra_state:
            checkpoint.update(extra_state)

        checkpoint = _to_cpu(checkpoint)

        path = os.path.join(checkpoint_path, f"checkpoint_step_{step}_rank_{rank}.pt")

        if call_wrapper is not None:
            with call_wrapper.atomic():
                torch.save(checkpoint, path)
        else:
            torch.save(checkpoint, path)

        del checkpoint

    gc.collect()
    torch.cuda.empty_cache()

    save_time = time.time() - start_time
    return path, save_time


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer,
    rank: int,
    strategy: str,
    extra_loaders: dict | None = None,
):
    """Load a training checkpoint if one exists.

    For FSDP, uses ``torch.distributed.checkpoint`` (DCP) to load sharded
    state dicts from a ``dcp_step_<N>/`` directory.  DCP handles the
    ShardedTensor/DTensor deserialization natively.

    For DDP, loads from per-rank ``checkpoint_step_<N>_rank_<R>.pt`` files
    or the legacy ``latest_checkpoint.pt``.

    Handles DDP ``module.`` prefix mismatches transparently.

    Args:
        checkpoint_path: Directory containing checkpoint files.
        model: Wrapped model.
        optimizer: Optimizer.
        rank: Global rank.
        strategy: ``"fsdp"`` or ``"ddp"``.
        extra_loaders: Optional dict of ``{key: callback(value)}`` for
            restoring additional state (e.g. ft_client).

    Returns:
        (start_step, extra_values) -- *extra_values* is a dict of the raw
        checkpoint values for each key in *extra_loaders* (or empty dict).
    """
    extra_values = {}

    if strategy == "fsdp":
        # Find the latest DCP checkpoint directory
        latest_step = _find_latest_dcp_checkpoint(checkpoint_path)
        if latest_step < 0:
            return 0, extra_values

        ckpt_dir = os.path.join(checkpoint_path, f"dcp_step_{latest_step}")

        try:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import (
                get_state_dict,
                set_state_dict,
                StateDictOptions,
            )

            # Get empty state dicts with the correct structure for DCP to fill.
            model_state, optim_state = get_state_dict(
                model,
                optimizer,
                options=StateDictOptions(full_state_dict=False, cpu_offload=True),
            )

            state_dict = {
                "model": model_state,
                "optimizer": optim_state,
            }

            # DCP load is collective -- all ranks must call.
            dcp.load(state_dict, checkpoint_id=ckpt_dir)

            # Apply the loaded state dicts back to model and optimizer.
            set_state_dict(
                model,
                optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optimizer"],
                options=StateDictOptions(full_state_dict=False),
            )

            # Load extra state (step number, ft_client, etc.)
            extra_path = os.path.join(checkpoint_path, f"extra_step_{latest_step}.pt")
            start_step = latest_step
            if os.path.exists(extra_path):
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                extra = torch.load(extra_path, map_location=f"cuda:{local_rank}")
                start_step = extra.get("step", latest_step)

                if extra_loaders:
                    for key, loader_fn in extra_loaders.items():
                        if key in extra:
                            loader_fn(extra[key])
                            extra_values[key] = extra[key]

            logger.info(
                f"Loaded DCP checkpoint from step {start_step} "
                f"(dir={os.path.basename(ckpt_dir)})"
            )
            return start_step, extra_values

        except Exception as e:
            logger.warning(f"Failed to load DCP checkpoint: {e}")
            return 0, extra_values

    else:
        # DDP: load from per-rank torch.save files
        import glob as glob_module

        pattern = os.path.join(checkpoint_path, f"checkpoint_step_*_rank_{rank}.pt")
        rank_files = sorted(glob_module.glob(pattern))
        if rank_files:
            path = rank_files[-1]
        else:
            path = os.path.join(checkpoint_path, "latest_checkpoint.pt")

        if not os.path.exists(path):
            return 0, extra_values

        try:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            ckpt = torch.load(path, map_location=f"cuda:{local_rank}")

            saved_state = ckpt["model_state_dict"]

            # Handle DDP module. prefix mismatch
            model_keys = set(model.state_dict().keys())
            saved_keys = set(saved_state.keys())
            if model_keys and saved_keys and model_keys != saved_keys:
                if all(k.startswith("module.") for k in model_keys) and not any(
                    k.startswith("module.") for k in saved_keys
                ):
                    saved_state = {"module." + k: v for k, v in saved_state.items()}
                elif not any(k.startswith("module.") for k in model_keys) and all(
                    k.startswith("module.") for k in saved_keys
                ):
                    saved_state = {
                        k.replace("module.", "", 1): v for k, v in saved_state.items()
                    }

            model.load_state_dict(saved_state)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

            if extra_loaders:
                for key, loader_fn in extra_loaders.items():
                    if key in ckpt:
                        loader_fn(ckpt[key])
                        extra_values[key] = ckpt[key]

            start_step = ckpt["step"]
            logger.info(
                f"Loaded checkpoint from step {start_step} "
                f"(strategy={ckpt.get('parallel_strategy', 'unknown')}, "
                f"file={os.path.basename(path)})"
            )
            return start_step, extra_values

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return 0, extra_values
