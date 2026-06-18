#!/usr/bin/env python3
"""Qwen3-8B Pre-Training on H200 — Megatron-Core (NeMo 25.07)

Uses the Megatron pretrain API with Transformer Engine spec.
Qwen3-8B dimensions mapped to GPT model provider (Qwen3 bridge not registered).

Best config: TP=1, PP=1, DP=16, MBS=2, GBS=128, seq=4096, BF16
Result: 497 TFLOP/s/GPU, 162K tok/s on 16x H200
"""
import os
import sys

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

from functools import partial

import torch
from megatron.training import get_args, pretrain
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset


def model_provider(pre_process=True, post_process=True):
    """Build GPT model with Qwen3-8B dimensions."""
    args = get_args()
    config = core_transformer_config_from_args(args)

    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec()

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=True,
    )
    return model


def forward_step(data_iterator, model):
    """Forward pass — standard GPT causal LM loss."""
    args = get_args()
    tokens, labels, loss_mask, attention_mask, position_ids = _get_batch(data_iterator)
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, partial(_loss_func, loss_mask)


def _loss_func(loss_mask, output_tensor):
    """Compute averaged cross-entropy loss."""
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss, {"lm loss": loss}


def _get_batch(data_iterator):
    """Get batch from data iterator."""
    args = get_args()
    data = next(data_iterator)

    tokens = data["tokens"].long().cuda()
    labels = data["labels"].long().cuda()
    loss_mask = data["loss_mask"].float().cuda()
    attention_mask = data["attention_mask"].long().cuda() if "attention_mask" in data else None
    position_ids = data["position_ids"].long().cuda()

    return tokens, labels, loss_mask, attention_mask, position_ids


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build mock datasets for benchmarking.

    To switch to real data, replace MockGPTDataset with GPTDataset and provide:
      --data-path /fsx/ubuntu/qwen3-8b-pretraining/datasets/c4/merged_text_document
      --tokenizer-type HuggingFaceTokenizer
      --tokenizer-model Qwen/Qwen3-8B
    """
    args = get_args()

    config = GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        mock=True,
        mock_seq_length=args.seq_length,
    )

    dataset_builder = BlendedMegatronDatasetBuilder(
        MockGPTDataset, train_val_test_num_samples, lambda: True, config
    )
    train_ds, valid_ds, test_ds = dataset_builder.build()
    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={
            # Qwen3-8B architecture
            "num_layers": 36,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "group_query_attention": True,
            "num_query_groups": 8,
            "ffn_hidden_size": 14336,
            "swiglu": True,
            "max_position_embeddings": 4096,
            "seq_length": 4096,
            "padded_vocab_size": 151936,
            "use_rotary_position_embeddings": True,
            "rotary_percent": 1.0,
            "normalization": "RMSNorm",
            "untie_embeddings_and_output_weights": True,
            # Training
            "micro_batch_size": 2,
            "global_batch_size": 128,
            "train_iters": 100,
            "lr": 3e-4,
            "min_lr": 3e-5,
            "lr_warmup_iters": 10,
            "lr_decay_iters": 100,
            "lr_decay_style": "cosine",
            "weight_decay": 0.1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "clip_grad": 1.0,
            "bf16": True,
            # Parallelism
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "use_distributed_optimizer": True,
            "overlap_grad_reduce": True,
            "overlap_param_gather": True,
            # Gradient checkpointing (mandatory on H200 for MBS>=2)
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
            # Logging
            "log_interval": 5,
            "eval_interval": 1000,
            "eval_iters": 0,
            "tensorboard_dir": "/fsx/ubuntu/qwen3-8b-pretraining/tensorboard/h200",
            "save": "/fsx/ubuntu/qwen3-8b-pretraining/checkpoints/h200",
            "save_interval": 1000,
            "tokenizer_type": "NullTokenizer",
            "vocab_size": 151936,
        },
    )
