"""
Configuration classes for QLoRA fine-tuning of Qwen3-8B.

This module defines dataclasses for all configurable parameters used
in the fine-tuning process, including model settings, QLoRA parameters,
dataset configuration, and training hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import yaml


@dataclass
class ParallelConfig:
    """
    Configuration for distributed training parallelism.

    Supports Data Parallelism (DDP) and optimizer/gradient sharding
    via DeepSpeed ZeRO to reduce per-GPU memory usage.

    Attributes:
        strategy: Parallelism strategy to use:
            - "auto": Use DDP if >1 GPU, single-GPU otherwise
            - "ddp": PyTorch DistributedDataParallel only
            - "deepspeed_zero2": DDP + optimizer/gradient sharding (recommended for QLoRA)
            - "deepspeed_zero3": DDP + full parameter sharding (maximum memory savings)
        num_gpus: Number of GPUs to use (0 = auto-detect)
        deepspeed_config_path: Path to DeepSpeed JSON config file
        gradient_checkpointing: Enable gradient checkpointing to trade compute for memory
    """
    strategy: str = "deepspeed_zero2"  # Best balance for QLoRA multi-GPU
    num_gpus: int = 0  # 0 = auto-detect available GPUs
    deepspeed_config_path: Optional[str] = None
    gradient_checkpointing: bool = True


@dataclass
class ModelConfig:
    """
    Configuration for the base model.

    Attributes:
        model_name: Hugging Face model identifier (default: Qwen/Qwen3-8B)
        max_seq_length: Maximum sequence length for training (default: 2048)
        torch_dtype: Data type for model weights (default: bfloat16)
        trust_remote_code: Whether to trust remote code from HF Hub
    """
    model_name: str = "Qwen/Qwen3-8B"
    max_seq_length: int = 2048
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True


@dataclass
class QLoRAConfig:
    """
    Configuration for QLoRA (Quantized Low-Rank Adaptation).

    QLoRA enables efficient fine-tuning by:
    1. Quantizing the base model to 4-bit precision
    2. Adding trainable low-rank adapters
    3. Using paged optimizers for memory efficiency

    Attributes:
        load_in_4bit: Enable 4-bit quantization
        bnb_4bit_quant_type: Quantization type ("nf4" recommended for LLMs)
        bnb_4bit_use_double_quant: Enable nested quantization for memory savings
        bnb_4bit_compute_dtype: Dtype for computations during forward pass
        lora_r: Rank of the low-rank matrices (higher = more capacity)
        lora_alpha: Scaling factor for LoRA weights
        lora_dropout: Dropout probability for LoRA layers
        target_modules: Model layers to apply LoRA to
        task_type: Type of task (CAUSAL_LM for language modeling)
        bias: Bias training strategy ("none", "all", or "lora_only")
    """
    # Quantization Settings (BitsAndBytes)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"  # NormalFloat4 - optimal for normally distributed weights
    bnb_4bit_use_double_quant: bool = True  # Saves ~0.4 bits per parameter
    bnb_4bit_compute_dtype: str = "bfloat16"  # Use bfloat16 for stability

    # LoRA Settings
    lora_r: int = 16  # Rank of the update matrices
    lora_alpha: int = 32  # Scaling factor (typically 2x rank)
    lora_dropout: float = 0.05  # Regularization
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",      # Query projection in attention
        "k_proj",      # Key projection in attention
        "v_proj",      # Value projection in attention
        "o_proj",      # Output projection in attention
        "gate_proj",   # Gate projection in MLP
        "up_proj",     # Up projection in MLP
        "down_proj",   # Down projection in MLP
    ])
    task_type: str = "CAUSAL_LM"
    bias: str = "none"  # Don't train biases


@dataclass
class DataConfig:
    """
    Configuration for dataset loading and preprocessing.

    Attributes:
        dataset_name: Hugging Face dataset identifier
        dataset_config: Dataset configuration/subset name
        train_split: Fraction of data for training (rest is validation)
        max_samples: Maximum samples to use (None for all)
        num_workers: Number of workers for data loading
    """
    dataset_name: str = "FreedomIntelligence/medical-o1-reasoning-SFT"
    dataset_config: str = "en"  # English subset
    train_split: float = 0.95
    max_samples: Optional[int] = None  # Use all samples
    num_workers: int = 4


@dataclass
class TrainingConfig:
    """
    Configuration for the training loop.

    These hyperparameters are optimized for QLoRA fine-tuning on a
    single GPU with 24GB VRAM (e.g., NVIDIA A10G).

    Attributes:
        output_dir: Directory for checkpoints and logs
        per_device_train_batch_size: Batch size per GPU
        gradient_accumulation_steps: Steps to accumulate before update
        num_train_epochs: Total training epochs
        learning_rate: Peak learning rate
        weight_decay: L2 regularization coefficient
        warmup_ratio: Fraction of steps for LR warmup
        lr_scheduler_type: Learning rate schedule type
        optim: Optimizer type (paged_adamw_8bit for memory efficiency)
        max_grad_norm: Gradient clipping threshold
        bf16: Use bfloat16 mixed precision
        fp16: Use float16 mixed precision (mutually exclusive with bf16)
        logging_steps: Steps between logging
        save_steps: Steps between checkpoints
        eval_steps: Steps between evaluations
        save_total_limit: Maximum checkpoints to keep
        group_by_length: Group similar length sequences
        report_to: Logging backend
        seed: Random seed for reproducibility
    """
    output_dir: str = "./outputs"
    per_device_train_batch_size: int = 1  # Reduced from 2 to prevent OOM at 99% VRAM
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4  # With 4 GPUs × 1 batch × 4 accum = effective 16
    num_train_epochs: int = 3
    learning_rate: float = 2e-4  # Good starting point for QLoRA
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"  # Memory-efficient optimizer
    max_grad_norm: float = 0.3  # Aggressive clipping for stability
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    group_by_length: bool = True  # Reduces padding overhead
    report_to: str = "tensorboard"
    seed: int = 42


@dataclass
class FullConfig:
    """
    Complete configuration combining all sub-configurations.

    This class provides convenient loading from YAML files and
    serves as the single source of truth for all settings.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    qlora: QLoRAConfig = field(default_factory=QLoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "FullConfig":
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            qlora=QLoRAConfig(**config_dict.get('qlora', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            parallel=ParallelConfig(**config_dict.get('parallel', {})),
        )

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        config_dict = {
            'model': {
                'model_name': self.model.model_name,
                'max_seq_length': self.model.max_seq_length,
                'torch_dtype': self.model.torch_dtype,
                'trust_remote_code': self.model.trust_remote_code,
            },
            'qlora': {
                'load_in_4bit': self.qlora.load_in_4bit,
                'bnb_4bit_quant_type': self.qlora.bnb_4bit_quant_type,
                'bnb_4bit_use_double_quant': self.qlora.bnb_4bit_use_double_quant,
                'bnb_4bit_compute_dtype': self.qlora.bnb_4bit_compute_dtype,
                'lora_r': self.qlora.lora_r,
                'lora_alpha': self.qlora.lora_alpha,
                'lora_dropout': self.qlora.lora_dropout,
                'target_modules': self.qlora.target_modules,
                'task_type': self.qlora.task_type,
                'bias': self.qlora.bias,
            },
            'data': {
                'dataset_name': self.data.dataset_name,
                'dataset_config': self.data.dataset_config,
                'train_split': self.data.train_split,
                'max_samples': self.data.max_samples,
                'num_workers': self.data.num_workers,
            },
            'training': {
                'output_dir': self.training.output_dir,
                'per_device_train_batch_size': self.training.per_device_train_batch_size,
                'per_device_eval_batch_size': self.training.per_device_eval_batch_size,
                'gradient_accumulation_steps': self.training.gradient_accumulation_steps,
                'num_train_epochs': self.training.num_train_epochs,
                'learning_rate': self.training.learning_rate,
                'weight_decay': self.training.weight_decay,
                'warmup_ratio': self.training.warmup_ratio,
                'lr_scheduler_type': self.training.lr_scheduler_type,
                'optim': self.training.optim,
                'max_grad_norm': self.training.max_grad_norm,
                'bf16': self.training.bf16,
                'fp16': self.training.fp16,
                'logging_steps': self.training.logging_steps,
                'save_steps': self.training.save_steps,
                'eval_steps': self.training.eval_steps,
                'save_total_limit': self.training.save_total_limit,
                'group_by_length': self.training.group_by_length,
                'report_to': self.training.report_to,
                'seed': self.training.seed,
            },
            'parallel': {
                'strategy': self.parallel.strategy,
                'num_gpus': self.parallel.num_gpus,
                'deepspeed_config_path': self.parallel.deepspeed_config_path,
                'gradient_checkpointing': self.parallel.gradient_checkpointing,
            }
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
