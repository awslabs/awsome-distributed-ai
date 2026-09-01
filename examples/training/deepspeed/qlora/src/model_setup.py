"""
Model setup with QLoRA configuration.

This module handles loading the Qwen3-8B model with 4-bit quantization
and configuring LoRA adapters for efficient fine-tuning.
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import logging

logger = logging.getLogger(__name__)


def log_gpu_memory(stage: str):
    """
    Log current GPU memory usage for a given stage of training.

    Useful for diagnosing memory pressure and identifying operations
    that cause memory spikes (e.g., checkpoint saves).

    Args:
        stage: Description of the current stage (e.g., "Before checkpoint save")
    """
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            pct = (reserved / total) * 100 if total > 0 else 0
            logger.info(
                f"[{stage}] GPU {i}: "
                f"Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, "
                f"Total={total:.2f}GB ({pct:.1f}% used)"
            )


def get_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
    bnb_4bit_compute_dtype: str = "bfloat16"
) -> BitsAndBytesConfig:
    """
    Create BitsAndBytes configuration for 4-bit quantization.

    4-bit quantization reduces model memory footprint by ~4x while
    maintaining most of the model's capabilities. The NF4 (NormalFloat4)
    data type is specifically designed for normally distributed weights.

    Args:
        load_in_4bit: Enable 4-bit quantization
        bnb_4bit_quant_type: Quantization type
            - "nf4": NormalFloat4 (recommended for LLMs)
            - "fp4": Float4
        bnb_4bit_use_double_quant: Nested quantization for extra memory savings
        bnb_4bit_compute_dtype: Data type for matrix multiplications

    Returns:
        BitsAndBytesConfig instance

    Example:
        >>> config = get_quantization_config()
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "Qwen/Qwen3-8B",
        ...     quantization_config=config
        ... )
    """
    # Convert string dtype to torch dtype
    _valid_dtypes = {"float16", "bfloat16", "float32"}
    if bnb_4bit_compute_dtype not in _valid_dtypes:
        raise ValueError(
            f"Invalid compute dtype '{bnb_4bit_compute_dtype}'. "
            f"Must be one of: {', '.join(sorted(_valid_dtypes))}"
        )
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    logger.info(f"Creating quantization config:")
    logger.info(f"  - 4-bit: {load_in_4bit}")
    logger.info(f"  - Quant type: {bnb_4bit_quant_type}")
    logger.info(f"  - Double quant: {bnb_4bit_use_double_quant}")
    logger.info(f"  - Compute dtype: {compute_dtype}")

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list = None,
    bias: str = "none",
) -> LoraConfig:
    """
    Create LoRA (Low-Rank Adaptation) configuration.

    LoRA adds trainable low-rank matrices to frozen model weights,
    enabling efficient fine-tuning with minimal parameters.

    The effective update is: W' = W + BA
    Where:
    - W is the frozen original weight (d x k)
    - B is a trainable matrix (d x r)
    - A is a trainable matrix (r x k)
    - r is the rank (much smaller than d, k)

    Args:
        r: Rank of the low-rank matrices (higher = more capacity)
        lora_alpha: Scaling factor (alpha/r scales the adapter output)
        lora_dropout: Dropout for regularization
        target_modules: Which layers to apply LoRA to
        bias: Bias training strategy

    Returns:
        LoraConfig instance
    """
    if target_modules is None:
        # Default target modules for Qwen3 architecture
        target_modules = [
            "q_proj",     # Query projection in self-attention
            "k_proj",     # Key projection in self-attention
            "v_proj",     # Value projection in self-attention
            "o_proj",     # Output projection in self-attention
            "gate_proj",  # Gate projection in SwiGLU MLP
            "up_proj",    # Up projection in SwiGLU MLP
            "down_proj"   # Down projection in SwiGLU MLP
        ]

    logger.info(f"Creating LoRA config:")
    logger.info(f"  - Rank (r): {r}")
    logger.info(f"  - Alpha: {lora_alpha}")
    logger.info(f"  - Dropout: {lora_dropout}")
    logger.info(f"  - Target modules: {target_modules}")

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
    )


def _get_device_map(strategy: str = "deepspeed_zero2"):
    """
    Determine the appropriate device_map for model loading.

    For DDP / DeepSpeed ZeRO-2: each process loads the full model on its
    local GPU. Using device_map="auto" with DDP would incorrectly spread
    the model across all GPUs within a single process.

    For DeepSpeed ZeRO-3: device_map must be None because ZeRO-3 shards
    model parameters across ranks. The Trainer's DeepSpeed integration
    handles placement automatically.

    Args:
        strategy: Parallelism strategy (e.g., "deepspeed_zero2", "deepspeed_zero3")

    Returns:
        Device map specification for from_pretrained()
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        if "zero3" in strategy.lower():
            # ZeRO-3 shards parameters across ranks — device_map must be None
            device_map = None
            logger.info("Multi-GPU ZeRO-3 mode: device_map=None (Trainer manages placement)")
        else:
            # ZeRO-2 / DDP: pin model to this rank's GPU
            device_map = {"": local_rank}
            logger.info(f"Multi-GPU DDP mode: loading model on GPU {local_rank}")
    else:
        # Single-GPU: auto placement
        device_map = "auto"
        logger.info("Single-GPU mode: using device_map='auto'")

    return device_map


def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen3-8B",
    quantization_config: BitsAndBytesConfig = None,
    max_seq_length: int = 2048,
    trust_remote_code: bool = True,
    strategy: str = "deepspeed_zero2",
):
    """
    Load the base model and tokenizer from Hugging Face.

    Handles both single-GPU and multi-GPU (DDP/DeepSpeed) scenarios.
    In DDP mode, each process loads the model on its assigned GPU.

    Args:
        model_name: Hugging Face model identifier
        quantization_config: BitsAndBytesConfig for quantization
        max_seq_length: Maximum sequence length
        trust_remote_code: Trust remote code from model repo

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        model_max_length=max_seq_length,
    )

    # Ensure pad token is set (required for batch training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Set padding side (right padding is standard for causal LM)
    tokenizer.padding_side = "right"

    logger.info(f"Loading model: {model_name}")
    logger.info("This may take several minutes for large models...")

    device_map = _get_device_map(strategy=strategy)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16 if not quantization_config else None,
        attn_implementation="eager",  # Use eager attention (flash_attn not installed)
    )

    logger.info(f"Model loaded successfully")
    logger.info(f"Model device: {model.device}")
    log_gpu_memory("After model load")

    return model, tokenizer


def prepare_model_for_training(model, lora_config: LoraConfig):
    """
    Prepare the quantized model for QLoRA training.

    This function:
    1. Prepares the model for k-bit training (handles quantized weights)
    2. Adds LoRA adapters to target modules
    3. Enables gradient checkpointing for memory efficiency

    Args:
        model: Quantized base model
        lora_config: LoRA configuration

    Returns:
        Model with LoRA adapters
    """
    logger.info("Preparing model for k-bit training...")

    # Prepare for training with quantized weights
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )

    logger.info("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model


def setup_model_and_tokenizer(config):
    """
    Complete setup: load model with quantization and add LoRA adapters.

    This is the main entry point for model setup, combining all steps:
    1. Create quantization config
    2. Load model and tokenizer
    3. Add LoRA adapters

    Args:
        config: FullConfig instance with all settings

    Returns:
        Tuple of (model_with_lora, tokenizer)

    Example:
        >>> from config import FullConfig
        >>> config = FullConfig.from_yaml("configs/training_config_zero2.yaml")
        >>> model, tokenizer = setup_model_and_tokenizer(config)
    """
    logger.info("="*60)
    logger.info("Setting up model with QLoRA")
    logger.info("="*60)

    # Step 1: Create quantization config
    quant_config = get_quantization_config(
        load_in_4bit=config.qlora.load_in_4bit,
        bnb_4bit_quant_type=config.qlora.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.qlora.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=config.qlora.bnb_4bit_compute_dtype,
    )

    # Step 2: Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model.model_name,
        quantization_config=quant_config,
        max_seq_length=config.model.max_seq_length,
        trust_remote_code=config.model.trust_remote_code,
        strategy=config.parallel.strategy,
    )

    # Step 3: Create LoRA config
    lora_config = get_lora_config(
        r=config.qlora.lora_r,
        lora_alpha=config.qlora.lora_alpha,
        lora_dropout=config.qlora.lora_dropout,
        target_modules=config.qlora.target_modules,
        bias=config.qlora.bias,
    )

    # Step 4: Prepare model with LoRA
    model = prepare_model_for_training(model, lora_config)

    logger.info("="*60)
    logger.info("Model setup complete!")
    logger.info("="*60)

    return model, tokenizer


def get_gpu_memory_info():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            allocated = torch.cuda.memory_allocated(i) / 1e9

            logger.info(f"GPU {i} ({props.name}):")
            logger.info(f"  Total: {total:.1f} GB")
            logger.info(f"  Reserved: {reserved:.1f} GB")
            logger.info(f"  Allocated: {allocated:.1f} GB")
            logger.info(f"  Free: {total - reserved:.1f} GB")
    else:
        logger.warning("No CUDA devices available")
