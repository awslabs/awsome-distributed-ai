#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Thin launcher for V-JEPA 2 training via srun.

This script loads a YAML config and calls app.vjepa.train.main() directly,
which reads SLURM_LOCALID, SLURM_NTASKS, and SLURM_PROCID from the
environment to configure CUDA device selection and torch.distributed.

Why not use `python -m app.main --devices cuda:0`?
    app/main.py spawns a subprocess that passes rank_and_world_size=(0, 1) to
    init_distributed(), bypassing SLURM env vars. This causes each process to
    see world_size=1 instead of the actual SLURM world size. Calling
    app.vjepa.train.main() directly avoids this issue.

Environment variables for optional optimizations:
    VJEPA_FUSED_OPTIMIZER=1   - Use fused AdamW (single kernel for optimizer step)
    VJEPA_TF32=1              - Enable TF32 for float32 matmuls (free perf on Ampere+)
    VJEPA_COMPILE_MODE=<mode> - torch.compile mode: default, reduce-overhead, max-autotune
    VJEPA_GRAD_BUCKET_VIEW=1  - Enable gradient_as_bucket_view on DDP wrappers
    VJEPA_PREFETCH_FACTOR=<n> - DataLoader prefetch_factor (default: PyTorch default of 2)

Usage with srun:
    srun --ntasks-per-node=8 ... python scripts/run_train.py \
        --fname /path/to/config.yaml
"""

import argparse
import os
import pprint

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, required=True, help="Path to YAML config file")

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.fname, "r") as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(params)

    # -- Optimization: TF32 for float32 matmuls (Ampere+/Blackwell).
    # Operations outside BF16 autocast scope (LayerNorm, loss, optimizer state)
    # run in float32. TF32 uses Tensor Cores for these with no accuracy loss
    # for training workloads.
    if os.environ.get("VJEPA_TF32") == "1":
        import torch

        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[run_train] TF32 enabled for float32 matmuls")

    # -- Optimization: disable GradScaler for BF16 training.
    # BF16 has the same dynamic range as FP32, so loss scaling is unnecessary.
    # V-JEPA 2 unconditionally creates a GradScaler; monkey-patching it to
    # a no-op removes the scale/unscale/step/update overhead per iteration.
    # We subclass instead of using a lambda so that Apex's GradScaler (which
    # inherits from torch.cuda.amp.GradScaler) still works.
    if params.get("meta", {}).get("dtype") == "bfloat16":
        import torch.cuda.amp

        _OrigGradScaler = torch.cuda.amp.GradScaler

        class _DisabledGradScaler(_OrigGradScaler):
            def __init__(self, *args, **kwargs):
                kwargs["enabled"] = False
                super().__init__(*args, **kwargs)

        torch.cuda.amp.GradScaler = _DisabledGradScaler

    # -- Optimization: fused AdamW optimizer.
    # Fuses the entire optimizer step into a single CUDA kernel, reducing
    # memory bandwidth by avoiding separate reads for params/grads/state.
    if os.environ.get("VJEPA_FUSED_OPTIMIZER") == "1":
        import torch.optim

        _OrigAdamW = torch.optim.AdamW

        class _FusedAdamW(_OrigAdamW):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("fused", True)
                super().__init__(*args, **kwargs)

        torch.optim.AdamW = _FusedAdamW
        print("[run_train] Fused AdamW enabled")

    # -- Optimization: torch.compile mode override.
    # The upstream code calls model.compile() with no args (mode="default").
    # max-autotune enables kernel autotuning + internal CUDA graph capture.
    compile_mode = os.environ.get("VJEPA_COMPILE_MODE")
    if compile_mode:
        import torch

        _orig_compile = torch.compile

        def _patched_compile(model=None, **kwargs):
            kwargs.setdefault("mode", compile_mode)
            print(f"[run_train] torch.compile with mode={kwargs['mode']}")
            return _orig_compile(model, **kwargs)

        torch.compile = _patched_compile

    # -- Optimization: gradient_as_bucket_view for DDP.
    # Avoids an extra gradient copy during DDP bucketing.
    if os.environ.get("VJEPA_GRAD_BUCKET_VIEW") == "1":
        from torch.nn.parallel import DistributedDataParallel as _OrigDDP

        class _PatchedDDP(_OrigDDP):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("gradient_as_bucket_view", True)
                super().__init__(*args, **kwargs)

        import torch.nn.parallel

        torch.nn.parallel.DistributedDataParallel = _PatchedDDP
        # Also patch the direct import path used by upstream code
        import torch.nn.parallel as _tnp

        _tnp.DistributedDataParallel = _PatchedDDP
        print("[run_train] gradient_as_bucket_view=True enabled for DDP")

    # -- Optimization: DataLoader prefetch_factor override.
    prefetch = os.environ.get("VJEPA_PREFETCH_FACTOR")
    if prefetch:
        import torch.utils.data

        _OrigDataLoader = torch.utils.data.DataLoader

        class _PatchedDataLoader(_OrigDataLoader):
            def __init__(self, *args, **kwargs):
                if kwargs.get("num_workers", 0) > 0:
                    kwargs.setdefault("prefetch_factor", int(prefetch))
                super().__init__(*args, **kwargs)

        torch.utils.data.DataLoader = _PatchedDataLoader
        print(f"[run_train] DataLoader prefetch_factor={prefetch}")

    # Import train module - this triggers CUDA_VISIBLE_DEVICES setup from SLURM_LOCALID
    from app.vjepa.train import main as train_main

    train_main(args=params, resume_preempt=False)
