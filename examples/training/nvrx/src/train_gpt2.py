#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from NVRx's sections example: HF training and rank-local checkpoint recovery.
"""HF GPT-2 training with NVRx worker restart and checkpoint recovery.

Start in a nonexistent --run-dir. Use --resume to require its last completed checkpoint on a later invocation. NVRx restarts automatically require that checkpoint too; a failure before the first save fails rather than silently starting over. There is no allocation recovery.
"""
import argparse
import datetime
import logging
import os
import random
import signal
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

import training_checkpoint as checkpoint

logger = logging.getLogger(__name__)


def batch_indices(cursor, rank, world, batch_size, length):
    start = cursor * world * batch_size + rank * batch_size
    return [(start + index) % length for index in range(batch_size)]


def validate_tokens(tokens):
    if not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.int64 or tokens.ndim != 2:
        raise ValueError("Tokens must be an int64 tensor [samples, sequence_length]")
    if tokens.shape[0] == 0 or tokens.shape[1] < 2:
        raise ValueError("Causal LM requires samples with at least two tokens")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Require the last completed checkpoint; never start fresh")
    parser.add_argument("--tiny", action="store_true", help="Fixed small random GPT-2 and tokens; no download or preprocessing")
    parser.add_argument("--model-path", type=Path, help="Local GPT-2 save_pretrained directory, kept unchanged")
    parser.add_argument("--tokens-path", type=Path, help="Local torch.save int64 tensor, kept unchanged")
    parser.add_argument("--strategy", choices=["ddp", "fsdp"], default="ddp")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--checkpoint-interval", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--collective-timeout", type=int, default=120)
    parser.add_argument("--fault-step", type=int, help="Test only, initial attempt: kill after this update or stop before it")
    parser.add_argument("--fault-kind", choices=["kill", "stop"], default="kill")
    parser.add_argument("--fault-rank", type=int, default=1)
    parser.add_argument("--no-nvrx", action="store_true", help="Explicit torchrun control: disable monitor notifications only")
    args = parser.parse_args()
    if args.tiny == bool(args.model_path or args.tokens_path) or (not args.tiny and not (args.model_path and args.tokens_path)):
        parser.error("Choose --tiny OR both --model-path and --tokens-path")
    if min(args.max_steps, args.checkpoint_interval, args.batch_size, args.collective_timeout) <= 0 or args.learning_rate <= 0:
        parser.error("Steps, interval, batch size, timeout and learning rate must be positive")
    if args.device == "cpu" and (args.strategy != "ddp" or args.dtype != "float32"):
        parser.error("CPU supports DDP/float32 only")
    if args.fault_step is not None and not 1 <= args.fault_step <= args.max_steps:
        parser.error("--fault-step must be within the training updates")
    return args


def main():
    args = parse_args()
    rank, local_rank, world = (int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    restart = int(os.environ["TORCHELASTIC_RESTART_COUNT"])
    if args.fault_step is not None and not 0 <= args.fault_rank < world:
        raise ValueError("Fault rank must be in the training world")
    if args.no_nvrx and ("TORCHELASTIC_RUN_ID" not in os.environ or
                         "FT_RANK_MONITOR_IPC_SOCKET" in os.environ):
        raise ValueError("--no-nvrx requires torchrun, not ft_launcher")
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [Rank {rank}] %(message)s", force=True)
    device = torch.device("cuda", local_rank) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("bfloat16 is unsupported on this GPU")
    ft = None
    if not args.no_nvrx:
        from nvidia_resiliency_ext.fault_tolerance import RankMonitorClient
        ft = RankMonitorClient()
        ft.init_workload_monitoring()
        ft.start_section("setup")
    try:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo",
                                timeout=datetime.timedelta(seconds=args.collective_timeout))
        selected = checkpoint.select_checkpoint(args.run_dir, args.resume, restart)
        # All FSDP shards must originate from the same initialized model.
        torch.manual_seed(args.seed)
        if args.tiny:
            model = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=32, n_embd=32,
                                             n_layer=2, n_head=2, attn_pdrop=0.1,
                                             resid_pdrop=0.1, embd_pdrop=0.1))
            tokens = torch.randint(0, 64, (32, 16), generator=torch.Generator().manual_seed(42))
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
                                                       torch_dtype=getattr(torch, args.dtype))
            tokens = torch.load(args.tokens_path, map_location="cpu", weights_only=True)
        validate_tokens(tokens)
        if not isinstance(model, GPT2LMHeadModel):
            raise ValueError("This reference supports GPT2LMHeadModel only")
        if tokens.min() < 0 or tokens.max() >= model.config.vocab_size or tokens.shape[1] > model.config.n_positions:
            raise ValueError("Tokens exceed the model vocabulary or context length")
        model = model.to(device=device, dtype=getattr(torch, args.dtype))
        if any(not p.requires_grad or p.dtype != getattr(torch, args.dtype) for p in model.parameters()):
            raise ValueError("Expected uniform precision and all parameters trainable")
        if args.strategy == "fsdp":
            model = FSDP(model, device_id=device)
        else:
            model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        settings = {
            "world_size": world, "local_world_size": int(os.environ["LOCAL_WORLD_SIZE"]),
            "strategy": args.strategy, "dtype": args.dtype, "device": args.device,
            "batch_size": args.batch_size, "seed": args.seed, "learning_rate": args.learning_rate,
            "model_path": str(args.model_path.resolve()) if args.model_path else None,
            "tokens_path": str(args.tokens_path.resolve()) if args.tokens_path else None,
            "token_shape": list(tokens.shape), "tiny": args.tiny,
        }
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        step = cursor = 0
        if selected is not None:
            cursor = checkpoint.load(args.run_dir, selected, model, optimizer, scheduler, settings)
            step = selected
            logger.info("LOADED step=%s cursor=%s restart=%s", step, cursor, restart)
        else:
            logger.info("FRESH step=0 restart=%s", restart)
        if ft is not None:
            ft.end_section("setup")
        model.train()
        while step < args.max_steps:
            if ft is not None:
                ft.start_section("step")
            if args.fault_kind == "stop" and restart == 0 and step + 1 == args.fault_step and rank == args.fault_rank:
                logger.warning("INJECT SIGSTOP before step=%s rank=%s pid=%s", step + 1, rank, os.getpid())
                os.kill(os.getpid(), signal.SIGSTOP)
            indices = batch_indices(cursor, rank, world, args.batch_size, len(tokens))
            batch = tokens[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=batch, labels=batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite causal-LM loss")
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            cursor += 1
            logger.info("STEP step=%s samples=%s loss=%.8f", step, indices, loss.item())
            if args.fault_kind == "kill" and restart == 0 and step == args.fault_step and rank == args.fault_rank:
                logger.warning("INJECT SIGKILL step=%s rank=%s", step, rank)
                os.kill(os.getpid(), signal.SIGKILL)
            if ft is not None:
                ft.end_section("step")
            if step % args.checkpoint_interval == 0 or step == args.max_steps:
                if ft is not None:
                    ft.start_section("checkpointing")
                checkpoint.save(args.run_dir, model, optimizer, scheduler, step, cursor, settings)
                logger.info("SAVED step=%s", step)
                if ft is not None:
                    ft.end_section("checkpointing")
        logger.info("DONE step=%s", step)
    finally:
        if ft is not None:
            ft.shutdown_workload_monitoring()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
