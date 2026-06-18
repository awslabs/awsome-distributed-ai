#!/usr/bin/env python3
"""Qwen3-8B Pre-Training on B300 — Megatron-Bridge (NeMo 26.02)

Uses the Megatron-Bridge recipe API with config objects.
No gradient checkpointing needed (288 GB B300 memory).

Best config: TP=1, PP=1, DP=16, MBS=4, GBS=128, seq=4096, BF16
Result: 976 TFLOP/s/GPU, 318K tok/s on 16x B300
"""
import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from megatron.bridge.recipes.qwen.qwen3 import qwen3_8b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


def main():
    cfg = qwen3_8b_pretrain_config(
        mock=True,  # Set to False and provide data_path for real training
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        micro_batch_size=4,
        global_batch_size=128,
        seq_length=4096,
        train_iters=100,
        lr_warmup_iters=10,
        lr_decay_iters=100,
    )

    # Training configuration
    cfg.train.bf16 = True
    cfg.train.use_distributed_optimizer = True
    cfg.train.overlap_grad_reduce = True
    cfg.train.overlap_param_gather = True

    # No gradient checkpointing (B300 has 288 GB — fits MBS=4 without recompute)
    cfg.train.recompute_granularity = None

    # Optimizer
    cfg.optimizer.lr = 3e-4
    cfg.optimizer.min_lr = 3e-5
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.clip_grad = 1.0

    # Logging and checkpoints
    cfg.logger.log_interval = 5
    cfg.train.eval_interval = 1000
    cfg.train.eval_iters = 0
    cfg.train.dir = "/fsx/ubuntu/qwen3-8b-pretraining/checkpoints/b300"
    cfg.train.save_interval = 1000

    # To use real data instead of mock:
    # cfg.data.mock = False
    # cfg.data.data_path = "/fsx/ubuntu/qwen3-8b-pretraining/datasets/c4/merged_text_document"
    # cfg.data.tokenizer_type = "HuggingFaceTokenizer"
    # cfg.data.tokenizer_model = "Qwen/Qwen3-8B"

    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
