# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""GPU-free import and source identity checks. Does not validate EFA training."""
import argparse
import ctypes
import hashlib
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import runpy


MCORE_BASE = "bb5dfd08f09ce06c5925af453fef06b3129f199d"
BRIDGE_BASE = "281f4bebfd78eb7cc9e8282c7929e99a9746b6cc"
INIT_FIX_HEAD = "9f7cd50a440b0d29625cfea9c58d55ebb4484b1f"
INIT_PATCH_SHA256 = "fddfeb657ccd878a0f3599a7528c7a413b2ca48604a729bbce5b7781a4eed48e"
PATCHED_SOURCE_SHA256 = "2bad1188729a5d06ed6bff10efe17cae672f14ff8fc29f71de40e3967544be43"
PATCHED_FILE = "megatron/core/transformer/moe/fused_a2a.py"


def verify_sources():
    """Require the pinned base plus exactly the reviewed initialization backport."""
    patch = Path("/opt/benchmark/mcore-pr5153-comm-init.patch")
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == INIT_PATCH_SHA256
    for root, pin, allowed in (
        ("/opt/upstream/Megatron-LM", MCORE_BASE, [PATCHED_FILE]),
        ("/opt/upstream/Megatron-Bridge", BRIDGE_BASE, []),
    ):
        actual = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        assert actual == pin, (actual, pin)
        changed = subprocess.check_output(
            ["git", "-C", root, "diff", "--name-only", "HEAD"], text=True,
        ).splitlines()
        assert changed == allowed, (root, changed, allowed)
    core = Path("/opt/upstream/Megatron-LM")
    assert hashlib.sha256((core / PATCHED_FILE).read_bytes()).hexdigest() == PATCHED_SOURCE_SHA256
    subprocess.run(["git", "-C", str(core), "apply", "--reverse", "--check", str(patch)], check=True)
    print("SOURCE_IDENTITY " + json.dumps({
        "mcore_base": MCORE_BASE, "bridge_base": BRIDGE_BASE,
        "variant": "upstream dev + PR 5153 communicator-init backport",
        "init_fix_head": INIT_FIX_HEAD, "patch_sha256": INIT_PATCH_SHA256,
        "patched_source_sha256": PATCHED_SOURCE_SHA256,
    }, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-stub", action="store_true", help="Explicit GPU-free import check using the toolkit driver stub")
    parser.add_argument("--config", action="store_true", help="Also construct both Kimi dev configurations from staged HF files")
    parser.add_argument("--source-only", action="store_true", help="Verify the pinned base and exact backport without importing CUDA")
    args = parser.parse_args()
    verify_sources()
    if args.source_only:
        return
    if args.cuda_stub:
        ctypes.CDLL("/usr/local/cuda/lib64/stubs/libcuda.so", mode=ctypes.RTLD_GLOBAL)
        print("CUDA_STUB_IMPORT_ONLY: no CUDA execution is verified", flush=True)

    import torch
    import deep_ep
    import megatron.core
    import megatron.bridge
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain
    from megatron.bridge.recipes.deepseek.deepseek_v3 import (
        deepseek_v3_pretrain_config_32nodes,
        set_deepseek_v3_pipeline_model_parallel_layout,
    )
    from megatron.core.transformer.moe import fused_a2a, token_dispatcher

    # Source commits were already asserted by verify_sources(); here only prove
    # the imported modules resolve to those verified trees.
    for module, root in (
        (megatron.core, "/opt/upstream/Megatron-LM"),
        (megatron.bridge, "/opt/upstream/Megatron-Bridge"),
    ):
        assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
        print(module.__name__, module.__file__, flush=True)
    assert hasattr(deep_ep, "ElasticBuffer"), deep_ep.__file__
    assert "874779c" in metadata.version("deep-ep"), metadata.version("deep-ep")
    assert Path(deep_ep.__file__).resolve().is_relative_to("/opt/venv"), deep_ep.__file__
    assert fused_a2a.HAVE_DEEP_EP_V2 and fused_a2a.deepepv2_dispatch is not None
    assert fused_a2a.ElasticBuffer is deep_ep.ElasticBuffer
    assert "all_reduce" in fused_a2a.get_elastic_buffer.__code__.co_names
    assert '"deepepv2"' in inspect.getsource(token_dispatcher.MoEFlexTokenDispatcher.__init__)
    assert "callbacks" in inspect.signature(pretrain).parameters
    assert all(callable(fn) for fn in (forward_step, deepseek_v3_pretrain_config_32nodes, set_deepseek_v3_pipeline_model_parallel_layout))
    for package in ("torch", "transformer-engine", "megatron-core", "megatron-bridge", "deep-ep", "nvidia-nccl-cu13"):
        print(package, metadata.version(package), flush=True)
    print("deep_ep", deep_ep.__file__, flush=True)
    version = ctypes.c_int()
    assert ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(version)) == 0
    assert version.value == 23102, version.value
    nccl_paths = {
        str(Path(line.split()[-1]).resolve())
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/libnccl.so" in line
    }
    expected_nccl = str(Path("/opt/nccl/current/lib/libnccl.so.2").resolve())
    assert nccl_paths == {expected_nccl}, nccl_paths
    print("NCCL runtime", version.value, "Torch build", torch.cuda.nccl.version(), "loaded", sorted(nccl_paths), flush=True)
    for path in ("/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so", "/opt/nccl/current/include/nccl_device.h"):
        assert Path(path).is_file(), path
    print("IMPORT_CHECK_OK (GPU/EFA execution not tested)", flush=True)
    if args.config:
        entrypoint = runpy.run_path("/opt/benchmark/bench_kimi_k2_pretrain.py")
        for arm in ("alltoall", "deepepv2"):
            os.environ["MOE_DISPATCHER"] = arm
            cfg = entrypoint["build_config"]()
            # Provider/Core validation is CPU-safe; ConfigContainer validation
            # additionally queries the GPU and is reserved for hardware runs.
            cfg.model.finalize()
            cfg.optimizer.finalize()
            assert cfg.model.moe_token_dispatcher_type == ("flex" if arm == "deepepv2" else "alltoall")
            assert cfg.model.moe_flex_dispatcher_backend == ("deepepv2" if arm == "deepepv2" else None)
            assert cfg.model.num_moe_experts == 384 and cfg.model.num_layers == 61
            assert cfg.model.mtp_num_layers is None
            assert not cfg.comm_overlap.overlap_moe_expert_parallel_comm
            assert cfg.optimizer.lr is not None and cfg.scheduler.lr_warmup_iters == 0
            print("CONFIG_CHECK_OK", arm, flush=True)


if __name__ == "__main__":
    main()
