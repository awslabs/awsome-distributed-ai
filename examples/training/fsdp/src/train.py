# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import functools
import math
import time

import torch
from torch import optim
import torch.distributed as dist
import torch.utils.data

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
from torch.utils.data import DataLoader

from model_utils.train_utils import (get_model_config, 
                                   compute_num_params,
                                   get_transformer_layer,
                                   get_learning_rate_scheduler,
                                   create_streaming_dataloader)
from model_utils.checkpoint import save_checkpoint, load_checkpoint
from model_utils.arguments import parse_args

import logging
import sys

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=logging.INFO, stream=sys.stdout)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def eval_model(model, dataloader, num_batches):
    """Eval step."""
    model = model.eval()
    n_batches = 0
    loss = 0.0

    with torch.no_grad():
        for batch_idx, input_data in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            loss += model(input_ids=input_data, attention_mask=None, labels=input_data)["loss"]
            n_batches += 1

    if n_batches > 0:
        detached_loss = loss.detach()
        torch.distributed.all_reduce(detached_loss)
        loss = detached_loss.item() / dist.get_world_size()
        loss /= n_batches
        ppl = math.exp(loss)
    else:
        loss = -1.0
        ppl = -1.0

    return loss, ppl

def train(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
        model_config,
        num_params,
        args,
        global_rank,
        world_size,
        total_steps=0,
        start_batch_index=0
    ):
    model.train()
    for index in range(args.epochs):
        for batch_idx, input_data in enumerate(train_dataloader):
            if batch_idx < start_batch_index:
                continue
            optimizer.zero_grad(set_to_none=True)
            step_start = time.time()
            loss = model(input_ids=input_data, attention_mask=None, labels=input_data)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            lr_scheduler.step()
            total_steps += 1
            loss_metric = loss.item()
            step_time = time.time() - step_start
            sample_processed = input_data.shape[0] * world_size
            throughput = sample_processed / step_time
            # Throughput in tokens/sec and per-GPU TFLOPS using the standard
            # Llama approximation 6 * num_params * tokens (forward+backward).
            tokens_per_step = sample_processed * input_data.shape[1]
            tokens_per_sec = tokens_per_step / step_time
            tflops_per_gpu = (6.0 * num_params * tokens_per_step) / (
                step_time * world_size * 1e12
            )
            mfu = tflops_per_gpu / args.peak_tflops_per_gpu
            loss_scalar = loss.item()
            current_lr = lr_scheduler.get_lr()
            if global_rank==0 and batch_idx%args.logging_freq==0:
                logger.info(
                    "Batch %d Loss: %.5f, Speed: %.2f samples/sec, "
                    "Tokens/s: %.0f, TFLOPS/GPU: %.1f, MFU: %.3f, lr: %.6f",
                    batch_idx,
                    loss_scalar,
                    throughput,
                    tokens_per_sec,
                    tflops_per_gpu,
                    mfu,
                    current_lr,
                )
            if args.validation_freq and not total_steps % args.validation_freq:
                val_loss, val_ppl = eval_model(
                    model, val_dataloader, args.validation_batches
                )
                model = model.train()
                if global_rank == 0:
                    logger.info(
                            "Batch %d Validation loss: %s",
                            batch_idx,
                            val_loss,
                        )
            if args.checkpoint_dir and not total_steps % args.checkpoint_freq:
                user_content = {
                    "cli_args": args.__dict__,
                    "num_params": num_params,
                    "total_steps": total_steps,
                    "model_config": model_config,
                    "start_batch_index": batch_idx + 1,
                }
                sub_dir = f"{args.model_type}-{total_steps}steps"

                save_checkpoint(
                    model,
                    optimizer,
                    lr_scheduler,
                    user_content,
                    args.checkpoint_dir,
                    sub_dir,
                )
            if total_steps >= args.max_steps:
                break
            

def main(args):
    dist.init_process_group()
    global_rank = dist.get_rank()
    device = global_rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    world_size = dist.get_world_size()
    
    if args.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.get_default_dtype()
    
    model_config = get_model_config(args)
    if global_rank == 0:
        logger.info("Creating Model with FSDP2")
    
    # Initialize model on meta device
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config)
    
    num_params = compute_num_params(model)
    if global_rank == 0:
        logger.info(
            "Created model with total parameters: %d (%.2f B)", num_params, num_params * 1e-9
        )
    
    transformer_layer = get_transformer_layer(args.model_type)

    # Configure FSDP2 options
    fsdp_kwargs = {}
    
    # Mixed precision policy
    if args.bf16:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    
    # Sharding strategy
    if args.sharding_strategy == "full":
        fsdp_kwargs["reshard_after_forward"] = True
    elif args.sharding_strategy == "hybrid":
        # For hybrid sharding, need 2D device mesh
        fsdp_kwargs["reshard_after_forward"] = True
    else:
        raise NotImplementedError("Available sharding strategies are full and hybrid")
    
    # CPU offload
    if args.cpu_offload == 1:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()
    
    # Apply fully_shard to transformer layers first
    for module in model.modules():
        if isinstance(module, transformer_layer):
            fully_shard(module, **fsdp_kwargs)
    
    # Apply fully_shard to root model
    fully_shard(model, **fsdp_kwargs)
    
    # Move model from meta device to CUDA
    model.to_empty(device=torch.device("cuda"))

    if global_rank == 0:
        logger.info("Wrapped model with FSDP2")

    if args.activation_checkpointing > 0:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
        check_fn = lambda submodule: isinstance(submodule, transformer_layer)
        wrapper_fn = functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=wrapper_fn, check_fn=check_fn
        )

    if args.offload_activations > 0:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper
        model = offload_wrapper(model)

    # Optimizer with DTensor parameters
    optimizer = optim.AdamW(
        model.parameters(), 
        betas=(args.beta1, args.beta2), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    if global_rank == 0:
        logger.info("Created optimizer")

    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    if args.resume_from_checkpoint:
        (
            model,
            optimizer,
            lr_scheduler,
            total_steps,
            start_batch_index,
        ) = load_checkpoint(model, 
                            optimizer, 
                            lr_scheduler, 
                            args.resume_from_checkpoint, 
                            args.model_type,
                            device)
    else:
        total_steps = 0
        start_batch_index = 0
    
    train_dataloader = create_streaming_dataloader(args.dataset, 
                                                   args.tokenizer, 
                                                   name=args.dataset_config_name, 
                                                   batch_size=args.train_batch_size, 
                                                   split='train')
    
    val_dataloader = create_streaming_dataloader(args.dataset, 
                                                  args.tokenizer, 
                                                  name=args.dataset_config_name, 
                                                  batch_size=args.val_batch_size, 
                                                  split='validation')
    
    train(model, 
          optimizer, 
          train_dataloader,
          val_dataloader,
          lr_scheduler, 
          model_config, 
          num_params, 
          args, 
          global_rank, 
          world_size,
          total_steps,
          start_batch_index)
  
    dist.destroy_process_group()

if __name__ == "__main__":
    args, _ = parse_args()
    main(args)

