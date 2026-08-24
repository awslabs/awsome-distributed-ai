#!/usr/bin/env bash
set -euo pipefail
backend="${1:?backend}"
out="${2:?output path}"
python3 - "${backend}" "${out}" <<'PY'
import ctypes
import hashlib
import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys

backend, output = sys.argv[1:]
import torch

def command(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

nccl_path = pathlib.Path("/opt/nccl/build/lib/libnccl.so.2").resolve()
nccl = ctypes.CDLL(str(nccl_path))
runtime = ctypes.c_int()
assert nccl.ncclGetVersion(ctypes.byref(runtime)) == 0
manifest = {
    "schema_version": 2,
    "backend": backend,
    "base_digest": "sha256:d8257dd0c4da714843aa1e371fe4dc8ee5a7972c247abbfd28d680bb197b6195",
    "bridge_commit": os.environ["BRIDGE_COMMIT"],
    "mcore_commit": os.environ["MCORE_COMMIT"],
    "torch": torch.__version__,
    "torch_nccl_build": list(torch.cuda.nccl.version()),
    "torch_cuda": torch.version.cuda,
    "nccl_tag": os.environ["NCCL_TAG"],
    "nccl_commit": os.environ["NCCL_COMMIT"],
    "nccl_runtime": runtime.value,
    "nccl_path": str(nccl_path),
    "nccl_sha256": sha(nccl_path),
    "efa_installer": os.environ["EFA_INSTALLER_VERSION"],
    "libfabric": command("fi_info", "--version"),
    "gdrcopy_tag": os.environ["GDRCOPY_TAG"],
    "nvcc": command("nvcc", "--version"),
}
try:
    manifest["transformer_engine"] = importlib.metadata.version("transformer-engine")
except importlib.metadata.PackageNotFoundError:
    manifest["transformer_engine"] = None
pathlib.Path(output).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

