#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Render one FSx-only, MB4 comparison job using the existing raw-Pod launcher.

This command does not contact Kubernetes. Stage the benchmark and measurement
entrypoints under --source-dir first, and keep rendered manifests outside Git.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import yaml

BACKENDS = {
    "nccl-alltoall": "alltoall",
    "uccl": "deepep",
    "deepep-v1-nvshmem": "deepep",
    "deepep-v2-gin-gda": "deepepv2",
}


def render(args):
    source = args.source_dir.rstrip("/")
    if not source.startswith("/fsx/") or not args.tmp_dir.startswith("/fsx/"):
        raise ValueError("source and temporary files must use the mounted FSx PVC")
    if len(args.tmp_dir) > 45:
        raise ValueError("use a short FSx temporary path for Unix-domain sockets")
    env = dict(os.environ)
    if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", env.get("IMG", "")):
        raise ValueError("IMG must be an immutable image digest")
    nodes = env.get("NODE_NAMES", "").split(",")
    if len(nodes) != 32 or len(set(nodes)) != 32 or not all(nodes):
        raise ValueError("NODE_NAMES must identify the assigned 32 distinct nodes")
    label = env.get("ARM_LABEL", f"cmp-{args.kind}-{args.arm}-r{args.repeat}")
    campaign = env["CAMPAIGN_ID"]
    env.update(
        STORAGE="pvc", MODEL="kimi-k2", RENDER_ONLY="1", ARM_LABEL=label,
        BENCH_PY=source + "/short_kimi_training_with_update_norm.py",
        TENSOR_PARALLEL="8", PIPELINE_PARALLEL="8", EXPERT_PARALLEL="32",
        MICRO_BATCH="4", GLOBAL_BATCH="256", SEQ_LEN="4096",
        TRAIN_ITERS="8" if args.kind == "correctness" else "40",
        MOE_A2A_OVERLAP="off", HF_HUB_OFFLINE="1",
    )
    launcher = Path(__file__).with_name("3.run-deepep-v2.sh")
    backend = BACKENDS[args.arm]
    result = subprocess.run(
        ["bash", str(launcher), "alltoall" if backend == "alltoall" else "deepepv2", "32"],
        env=env, text=True, capture_output=True, check=True,
    )
    sys.stderr.write(result.stderr)
    documents = list(yaml.safe_load_all(result.stdout))
    run_dir = f"/fsx/megatron-bridge-bench/{campaign}/kimi-k2/{label}-mb4-ovloff"
    for pod in documents:
        if pod["kind"] != "Pod":
            continue
        rank = int(pod["metadata"]["labels"]["rank"])
        container = pod["spec"]["containers"][0]
        command = container["args"][0]
        if command.count("torchrun ") != 1:
            raise ValueError("unexpected launcher layout")
        values = {
            "EP_ARM": args.arm, "COMPARISON_PRIMARY": "1",
            "COMPARISON_RUN_KIND": args.kind, "BENCHMARK_LEARNING_RATE": "0.000005",
            "KIMI_BENCHMARK_ENTRYPOINT": source + "/bench_kimi_k2_pretrain.py",
            "ROUTER_TRACE_DIR": f"{run_dir}/node-{rank}/routes",
            "ROUTER_TRACE_MAX_TRAINING_ITERS": "1",
            "EP_JIT_CACHE_DIR": f"{run_dir}/node-{rank}/jit-cache",
            "CUDA_CACHE_PATH": f"{run_dir}/node-{rank}/cuda-cache",
            "TRITON_CACHE_DIR": f"{run_dir}/node-{rank}/triton-cache",
            "TMPDIR": f"{args.tmp_dir}/{rank}",
            "NCCL_SYM_GIN_KERNELS_ENABLE": "0", "NCCL_GIN_TYPE": "5",
        }
        prefix = "mkdir -p " + shlex.quote(values["TMPDIR"]) + "; export "
        prefix += " ".join(key + "=" + shlex.quote(value) for key, value in values.items()) + "; "
        prefix += "python3 /opt/benchmark/verify_deepep_v2.py --source-only > "
        prefix += f"{run_dir}/logs/source-node-{rank}.log 2>&1 || exit $?; "
        prefix += f"sha256sum {source}/short_kimi_training_with_update_norm.py {source}/bench_kimi_k2_pretrain.py > {run_dir}/logs/source-node-{rank}.sha256; "
        # Rewrite the dispatcher toggle AND the env.txt receipt tokens so the
        # on-disk provenance names the arm that actually runs (the launcher
        # was invoked with its deepepv2 path for every flex arm).
        container["args"][0] = (
            command.replace("MOE_DISPATCHER=deepepv2", "MOE_DISPATCHER=" + backend)
            .replace(" arm=deepepv2 ", " arm=" + args.arm + " ")
            .replace("torchrun ", prefix + "torchrun ")
        )
        volumes = pod["spec"]["volumes"]
        if not any("persistentVolumeClaim" in volume for volume in volumes):
            raise ValueError("FSx PVC missing")
        if any(volume.get("hostPath", {}).get("type") != "CharDevice" for volume in volumes if "hostPath" in volume):
            raise ValueError("hostPath storage is forbidden")
    return documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=BACKENDS)
    parser.add_argument("--kind", choices=("correctness", "performance"), required=True)
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--source-dir", required=True, help="Staged entrypoints on the same FSx PVC")
    parser.add_argument("--tmp-dir", required=True, help="Fresh short FSx path, unique per job")
    args = parser.parse_args()
    sys.stdout.write(yaml.safe_dump_all(render(args), sort_keys=False))


if __name__ == "__main__":
    main()
