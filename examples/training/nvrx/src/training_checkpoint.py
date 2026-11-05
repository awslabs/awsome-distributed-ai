"""Synchronous PyTorch checkpoints; only the completed-step pointer selects a save.

DDP stores each logical rank's full state with torch.save. FSDP uses DCP for sharded model/optimizer state and torch.save for rank-local state. All ranks must call save/load together. Only load trusted checkpoints from this example with the same inputs and topology.
"""
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions

POINTER = "latest_checkpointed_iteration.txt"


def collective_device():
    """Explicit device for object collectives: the NCCL rank's current CUDA device, or CPU for Gloo.

    Gloo never queries CUDA, so the CPU test path stays clean on hosts without a GPU.
    """
    if dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def select_checkpoint(run_dir, resume, restart):
    """A new invocation requires a new directory; every restart requires a completed save."""
    root = Path(run_dir)
    selected: list[int | None] = [None]
    if dist.get_rank() == 0:
        if resume or restart > 0:
            step = int((root / POINTER).read_text())
            if step <= 0:
                raise ValueError("Resume requires a positive completed checkpoint step")
            selected[0] = step
        else:
            root.mkdir(parents=True, exist_ok=False)
    dist.broadcast_object_list(selected, src=0, device=collective_device())
    return selected[0]


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def save(root, model, optimizer, scheduler, step, cursor, settings):
    """Publish progress only after every rank has finished saving all its state."""
    path = Path(root) / f"step_{step}"
    path.mkdir(exist_ok=True)
    extra = {
        "step": step, "cursor": cursor, "rank": dist.get_rank(),
        "settings": settings, "scheduler": scheduler.state_dict(), "rng": rng_state(),
    }
    if settings["strategy"] == "fsdp":
        model_state, optim_state = get_state_dict(
            model, optimizer, options=StateDictOptions(cpu_offload=True))
        dcp.save({"model": model_state, "optimizer": optim_state}, checkpoint_id=path / "dcp")
    else:
        extra["model"] = model.state_dict()
        extra["optimizer"] = optimizer.state_dict()
    torch.save(extra, path / f"rank_{dist.get_rank()}.pt")
    # A failed writer cannot cross this barrier, so no partial save becomes latest.
    dist.barrier()
    if dist.get_rank() == 0:
        temporary = Path(root) / (POINTER + ".tmp")
        temporary.write_text(f"{step}\n")
        os.replace(temporary, Path(root) / POINTER)
    dist.barrier()
    return path


def load(root, step, model, optimizer, scheduler, settings):
    path = Path(root) / f"step_{step}"
    extra = torch.load(path / f"rank_{dist.get_rank()}.pt", map_location="cpu", weights_only=False)
    if extra["settings"] != settings:
        raise ValueError("Resume settings differ: keep the same inputs, precision and topology")
    if extra["rank"] != dist.get_rank() or extra["step"] != step or extra["cursor"] != step:
        raise ValueError("Inconsistent checkpoint rank/step/data cursor")
    if settings["strategy"] == "fsdp":
        model_state, optim_state = get_state_dict(
            model, optimizer, options=StateDictOptions(cpu_offload=True))
        state = {"model": model_state, "optimizer": optim_state}
        dcp.load(state, checkpoint_id=path / "dcp")
        set_state_dict(model, optimizer, model_state_dict=state["model"],
                       optim_state_dict=state["optimizer"])
    else:
        model.load_state_dict(extra["model"])
        if not extra["optimizer"]["state"]:
            raise ValueError("Completed checkpoint is missing optimizer state")
        optimizer.load_state_dict(extra["optimizer"])
    scheduler.load_state_dict(extra["scheduler"])
    restore_rng(extra["rng"])
    dist.barrier()
    return extra["cursor"]
