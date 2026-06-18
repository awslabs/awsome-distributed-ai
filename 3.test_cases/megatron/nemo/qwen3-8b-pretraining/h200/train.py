#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Qwen3-8B Pre-Training on H200 — Megatron-Bridge (NeMo 25.07)

Uses the Megatron-Bridge recipe API with config objects.
No gradient checkpointing needed (distributed optimizer keeps memory at ~114 GB).

Best config: TP=1, PP=1, DP=16, MBS=2, GBS=128, seq=4096, BF16
Result: 497 TFLOP/s/GPU, 162K tok/s on 16x H200
"""
import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from megatron.bridge.recipes.qwen.qwen3 import qwen3_8b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


def main():
    cfg = qwen3_8b_pretrain_config(
        mock=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        micro_batch_size=2,
        global_batch_size=128,
        seq_length=4096,
        train_iters=100,
        lr_warmup_iters=10,
        lr_decay_iters=100,
    )

    # No gradient checkpointing - MBS=2 with distributed optimizer fits in 141 GB
    # (uses ~114 GB peak, leaving headroom)

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
    cfg.train.dir = "/fsx/ubuntu/qwen3-8b/checkpoints/h200"
    cfg.train.save_interval = 1000

    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
