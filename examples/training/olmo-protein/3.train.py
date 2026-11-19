"""
Pretrain a protein language model with OLMo-core — MoE or a FLOP-matched dense
twin, selected by CLI flag. One token per amino-acid residue (28-token vocab,
no BPE); data shards from 2.prepare_uniref.py.

Launch with torchrun:

    torchrun --nproc-per-node=4 3.train.py run01 --arch moe   [KEY=VALUE overrides...]
    torchrun --nproc-per-node=4 3.train.py run01 --arch dense [KEY=VALUE overrides...]

Env:
    DATA_ROOT   dir or URL holding the tokenized .npy shards (default ./data/protein)
    SAVE_ROOT   checkpoint root (default ./checkpoints)
"""

import os
import sys
from dataclasses import dataclass
from glob import glob
from typing import List, cast

from olmo_core.config import Config, DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

SEQUENCE_LENGTH = 2048
DATA_ROOT = os.environ.get("DATA_ROOT", "./data/protein").rstrip("/")
SAVE_ROOT = os.environ.get("SAVE_ROOT", "./checkpoints")


def protein_tokenizer_config() -> TokenizerConfig:
    """28-token per-residue vocabulary built by 1.build_tokenizer.py:
    EOS=0, PAD=1, 20 canonical amino acids, 6 ambiguity codes (X B Z J U O)."""
    return TokenizerConfig(
        vocab_size=28,
        eos_token_id=0,
        pad_token_id=1,
        identifier="protein_aa_v1",
    )


def build_model_config(arch: str, tokenizer: TokenizerConfig) -> TransformerConfig:
    """MoE and dense twin, FLOP-matched: dense params ~= MoE *active* params.

    Measured with olmo-core 3.0.0 config math:
      MoE:   24 layers, d_model 2048, 8 experts top-2 -> 2.820B total / 1.008B active
      dense: identical trunk, FFN hidden_size_multiplier=0.75 -> 1.007B params
    (0.07% mismatch; padded_vocab_size rounds 28 up to 128 for GPU-friendly shapes)
    """
    vocab = tokenizer.padded_vocab_size()
    if arch == "moe-small":
        # Smoke-test variant (fits a single 24GB GPU): same shape family, tiny.
        return TransformerConfig.llama_like_moe(
            d_model=256,
            vocab_size=vocab,
            n_layers=2,
            n_heads=4,
            num_experts=8,
            top_k=2,
            expert_hidden_size=256,
            dropless=True,
            lb_loss_weight=0.01,
            z_loss_weight=0.001,
            reordered_norm=True,
            qk_norm=True,
            rope_theta=500_000,
        )
    elif arch == "dense-small":
        return TransformerConfig.llama_like(
            d_model=256,
            vocab_size=vocab,
            n_layers=2,
            n_heads=4,
            hidden_size_multiplier=0.75,
            qk_norm=True,
            rope_theta=500_000,
        )
    elif arch == "moe":
        return TransformerConfig.llama_like_moe(
            d_model=2048,
            vocab_size=vocab,
            n_layers=24,
            n_heads=16,
            num_experts=8,
            top_k=2,
            expert_hidden_size=2048,
            dropless=True,
            lb_loss_weight=0.01,
            z_loss_weight=0.001,
            reordered_norm=True,
            qk_norm=True,
            rope_theta=500_000,
        )
    elif arch == "dense":
        return TransformerConfig.llama_like(
            d_model=2048,
            vocab_size=vocab,
            n_layers=24,
            n_heads=16,
            # Sized so dense total params == MoE active params (1.008B): the
            # 0.75 multiplier lands at 1.007B. Equal compute per token.
            hidden_size_multiplier=0.75,
            qk_norm=True,
            rope_theta=500_000,
        )
    raise ValueError(f"--arch must be moe|dense|moe-small|dense-small, got {arch!r}")


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 6198  # same seed for both arms -> controlled comparison


def build_config(run_name: str, arch: str, overrides: List[str]) -> ExperimentConfig:
    tokenizer = protein_tokenizer_config()

    paths = sorted(glob(f"{DATA_ROOT}/train/*.npy")) or [f"{DATA_ROOT}/train/part-00-00000.npy"]

    dataset = NumpyFSLDatasetConfig(
        paths=paths,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=tokenizer,
        work_dir="/tmp/olmo-protein-dataset-cache",
    )

    data_loader = NumpyDataLoaderConfig(
        global_batch_size=64 * SEQUENCE_LENGTH,
        seed=6198,
        num_workers=4,
    )

    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=4 * SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=4e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            fused=True,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=2000),
    )

    trainer = (
        TrainerConfig(
            save_folder=f"{SAVE_ROOT}/{run_name}-{arch}",
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=5,
            # Step-based duration on purpose: epoch accounting can mislead
            # across resumes. Override on the CLI: trainer.max_duration.value=...
            max_duration=Duration.steps(11_584),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=200,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(name=f"{run_name}-{arch}", enabled=False),  # flip on to use W&B
        )
    )

    config = ExperimentConfig(
        model=build_model_config(arch, tokenizer),
        dataset=dataset,
        data_loader=data_loader,
        train_module=train_module,
        trainer=trainer,
    )
    return config.merge(overrides)


def main(run_name: str, arch: str, overrides: List[str]):
    config = build_config(run_name, arch, overrides)
    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config.as_config_dict()
    trainer.fit()


if __name__ == "__main__":
    usage = f"Usage: torchrun [OPTS..] {sys.argv[0]} run_name --arch moe|dense [OVERRIDES...]"
    if len(sys.argv) < 4 or sys.argv[2] != "--arch":
        print(usage)
        sys.exit(1)
    run_name, arch, overrides = sys.argv[1], sys.argv[3], sys.argv[4:]

    prepare_training_environment()
    try:
        main(run_name, arch, overrides)
    finally:
        teardown_training_environment()
