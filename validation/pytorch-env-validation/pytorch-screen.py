# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import torch

try:
    from rich.console import Console

    print = Console(force_terminal=True, force_jupyter=False).out
except ModuleNotFoundError:
    pass

print(f"{torch.cuda.is_available()=}")
print(f"{torch.version.cuda=}")
print(f"{torch.backends.cuda.is_built()=}")
print(f"{torch.backends.cuda.matmul.allow_tf32=}")
print(f"{torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=}")
print(f"{torch.backends.cuda.cufft_plan_cache=}")
print(f"{torch.backends.cuda.preferred_linalg_library(backend=None)=}")
print(f"{torch.backends.cuda.flash_sdp_enabled()=}")
print(f"{torch.backends.cuda.math_sdp_enabled()=}")

print(f"{torch.backends.cudnn.version()=}")
print(f"{torch.backends.cudnn.is_available()=}")
print(f"{torch.backends.cudnn.enabled=}")
print(f"{torch.backends.cudnn.allow_tf32=}")
print(f"{torch.backends.cudnn.deterministic=}")
print(f"{torch.backends.cudnn.benchmark=}")
print(f"{torch.backends.cudnn.benchmark_limit=}")

print(f"{torch.backends.mkl.is_available()=}")
print(f"{torch.backends.mkldnn.is_available()=}")

print(f"{torch.backends.openmp.is_available()=}")
try:
    print(f"{torch.backends.opt_einsum.is_available()=}")
    print(f"{torch.backends.opt_einsum.get_opt_einsum()=}")
    print(f"{torch.backends.opt_einsum.enabled=}")
    print(f"{torch.backends.opt_einsum.strategy=}")
except AttributeError:
    pass

print(f"{torch.distributed.is_available()=}")
print(f"{torch.distributed.is_mpi_available()=}")
print(f"{torch.distributed.is_nccl_available()=}")
