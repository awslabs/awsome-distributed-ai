# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import re
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
import torch.distributed.checkpoint as dist_cp
from model_utils.train_utils import get_logger

logger = get_logger()

def save_checkpoint(model, optimizer, scheduler, user_content, root_dir, sub_dir):
    """Save checkpoint using FSDP2 DTensor state dict APIs."""
    torch.cuda.empty_cache()

    save_dir = os.path.join(root_dir, sub_dir)
    if dist.get_rank() == 0:
        logger.info("Writing checkpoint to {0}.".format(save_dir))
    
    # Get sharded state dicts (DTensor format)
    model_state_dict = get_model_state_dict(model)
    optimizer_state_dict = get_optimizer_state_dict(model, optimizer)

    state_dict = {
        "model": model_state_dict,
        "optim": optimizer_state_dict,
        "scheduler": scheduler.state_dict(),
        "total_steps": user_content["total_steps"],
        "start_batch_index": user_content["start_batch_index"],
    }
    
    dist_cp.save(
        state_dict=state_dict,
        storage_writer=dist_cp.FileSystemWriter(save_dir),
    )
    
    dist.barrier()
    if dist.get_rank() == 0:
        logger.info("Completed checkpoint.")

def get_last_checkpoint(checkpoint_paths, model_type):
    """Find the most recent checkpoint."""
    steps = [int(re.findall(r'\d+steps', checkpoint.stem)[0].replace('steps','')) \
         for checkpoint in checkpoint_paths]
    checkpoints = sorted([(step, path) for step,path in zip(steps, checkpoint_paths)])
    
    # find last checkpoint, skipping incomplete ones 
    for step, path in reversed(checkpoints):
        metadata_path = path.joinpath(".metadata")
        if not metadata_path.exists():
            logger.warn(f"{metadata_path} not found. Skipping this incomplete checkpoint")
            continue
        return path.as_posix()
    else:
        return None
    
def load_checkpoint(model, optimizer, scheduler, checkpoint_dir, model_type, device):
    """Load checkpoint using FSDP2 DTensor state dict APIs."""
    checkpoint_paths = list(Path(checkpoint_dir).glob(f"{model_type}-*steps"))
    last_checkpoint = get_last_checkpoint(checkpoint_paths, model_type)
    
    if last_checkpoint is None:
        if dist.get_rank() == 0:
            logger.info("No Checkpoints Found")
        return(
            model,
            optimizer,
            scheduler,
            0,
            0,
        )
    
    if dist.get_rank() == 0:
        logger.info("Loading checkpoint from %s ...", last_checkpoint)
    
    # Load state dict from checkpoint
    state_dict = {
        "model": {},
        "optim": {},
        "scheduler": {},
        "total_steps": 0,
        "start_batch_index": 0,
    }
    
    dist_cp.load(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(last_checkpoint),
    )
    
    # Load model state dict
    set_model_state_dict(model, state_dict["model"])

    if dist.get_rank() == 0:
        logger.info("Loaded model state from disk")
        logger.info("Loading optimizer state from disk")
    
    # Load optimizer state dict
    set_optimizer_state_dict(model, optimizer, state_dict["optim"])

    # Load scheduler state
    scheduler.load_state_dict(state_dict["scheduler"])
    
    dist.barrier()
    if dist.get_rank() == 0:
        logger.info("Checkpoint loaded from %s.", last_checkpoint)
    
    return (
        model,
        optimizer,
        scheduler,
        state_dict["total_steps"],
        state_dict["start_batch_index"],
    )

