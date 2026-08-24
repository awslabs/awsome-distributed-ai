#!/usr/bin/env python3
"""Collect process-visible backend, NCCL, EFA, and topology provenance."""
from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import os
import pathlib
import subprocess
import sys


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"UNAVAILABLE: {exc}"


def main() -> None:
    destination = pathlib.Path(sys.argv[1])
    import torch

    identity = json.loads(pathlib.Path("/opt/benchmark/backend.json").read_text())
    requested = os.environ["EP_ARM"]
    if identity["ep_arm"] != requested:
        raise SystemExit(f"EP_ARM/image mismatch: {requested} != {identity}")

    deep_ep_path = None
    deep_ep_v2 = False
    if requested != "nccl-alltoall":
        deep_ep = importlib.import_module("deep_ep")
        deep_ep_path = str(pathlib.Path(deep_ep.__file__).resolve())
        deep_ep_v2 = hasattr(deep_ep, "ElasticBuffer")

    nccl_path = pathlib.Path("/opt/nccl/build/lib/libnccl.so.2").resolve()
    nccl = ctypes.CDLL(str(nccl_path))
    runtime = ctypes.c_int()
    if nccl.ncclGetVersion(ctypes.byref(runtime)) != 0:
        raise SystemExit("ncclGetVersion failed")
    maps = sorted({
        line.split()[-1]
        for line in pathlib.Path("/proc/self/maps").read_text().splitlines()
        if "libnccl.so" in line and "/" in line
    })
    resolved = sorted({str(pathlib.Path(item).resolve()) for item in maps})
    build_tuple = torch.cuda.nccl.version()
    build = build_tuple[0] * 10000 + build_tuple[1] * 100 + build_tuple[2]
    manifest = {
        "schema_version": 1,
        "ep_arm": requested,
        "backend_identity": identity,
        "environment": {name: os.environ.get(name) for name in (
            "NCCL_GIN_TYPE", "NCCL_SYM_GIN_KERNELS_ENABLE", "FI_PROVIDER",
            "FI_EFA_USE_DEVICE_RDMA", "NVSHMEM_REMOTE_TRANSPORT",
            "NVSHMEM_LIBFABRIC_PROVIDER", "PER_EXPERT_BATCHING")},
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_nccl_build": list(build_tuple),
        "nccl_runtime": runtime.value,
        "nccl_build_runtime_match": build == runtime.value,
        "nccl_path": str(nccl_path),
        "nccl_sha256": sha256(nccl_path),
        "loaded_nccl_paths": maps,
        "resolved_nccl_paths": resolved,
        "single_nccl": resolved == [str(nccl_path)],
        "deep_ep_path": deep_ep_path,
        "elastic_buffer_available": deep_ep_v2,
        "nvidia_smi": output(["nvidia-smi", "-q"]),
        "efa_devices": output(["fi_info", "-p", "efa"]),
        "rdma_links": output(["rdma", "link", "show"]),
    }
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"EP_IMAGE_IDENTITY_OK arm={requested}")
    print(f"NCCL_SINGLE_LIBRARY_OK value={str(manifest['single_nccl']).lower()} path={nccl_path}")
    print(f"NCCL_BUILD_RUNTIME_MATCH value={str(manifest['nccl_build_runtime_match']).lower()} build={build} runtime={runtime.value}")
    print(f"EFA_PROVIDER_PRESENT value={str('provider: efa' in manifest['efa_devices'].lower()).lower()}")
    if requested == "deepep-v2-gin-gda":
        print(f"DEEPEP_V2_IMPORT_OK ElasticBuffer={str(deep_ep_v2).lower()}")
        print(f"NCCL_GIN_TYPE_OK value={os.environ.get('NCCL_GIN_TYPE') == '5'}")


if __name__ == "__main__":
    main()
