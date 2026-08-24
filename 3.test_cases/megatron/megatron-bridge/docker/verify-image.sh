#!/usr/bin/env bash
set -euo pipefail
: "${EP_ARM:?}"
stub=/usr/local/cuda/lib64/stubs/libcuda.so
stub_link=""
if [[ -f "${stub}" && ! -e "${stub}.1" ]]; then
  stub_link="${stub}.1"
  ln -s "${stub}" "${stub_link}"
fi
trap '[[ -z "${stub_link}" ]] || rm -f "${stub_link}"' EXIT
export LD_LIBRARY_PATH="$(dirname "${stub}"):${LD_LIBRARY_PATH:-}"
python3 - "${EP_ARM}" <<'PY'
import ctypes
import importlib
import json
import os
import pathlib
import subprocess
import sys

arm = sys.argv[1]
identity_path = pathlib.Path("/opt/benchmark/backend.json")
identity = json.loads(identity_path.read_text())
assert identity["ep_arm"] == arm, (identity, arm)

import torch
from megatron.core.transformer.moe import token_dispatcher

assert hasattr(token_dispatcher, "_DeepepV2Manager")
if arm == "deepep-v2-gin-gda":
    import deep_ep
    assert hasattr(deep_ep, "ElasticBuffer")
elif arm in ("uccl", "deepep-v1-nvshmem"):
    import deep_ep
    assert hasattr(deep_ep, "Buffer") and not hasattr(deep_ep, "ElasticBuffer")

nccl_path = "/opt/nccl/build/lib/libnccl.so.2"
nccl = ctypes.CDLL(nccl_path)
runtime = ctypes.c_int()
assert nccl.ncclGetVersion(ctypes.byref(runtime)) == 0
assert runtime.value == 23102, runtime.value
build_tuple = torch.cuda.nccl.version()
build = build_tuple[0] * 10000 + build_tuple[1] * 100 + build_tuple[2]

mapped = sorted({
    line.split()[-1]
    for line in pathlib.Path("/proc/self/maps").read_text().splitlines()
    if "libnccl.so" in line and "/" in line
})
resolved = sorted({str(pathlib.Path(path).resolve()) for path in mapped})
assert resolved == [str(pathlib.Path(nccl_path).resolve())], (mapped, resolved)

extensions = [pathlib.Path(torch._C.__file__).resolve()]
for name in ("transformer_engine_torch", "deep_ep._C", "deep_ep_cpp", "uccl.ep"):
    try:
        module = importlib.import_module(name)
    except ImportError:
        continue
    path = pathlib.Path(module.__file__).resolve()
    if path.suffix == ".so":
        extensions.append(path)
    extensions.extend(path.parent.glob("*.so"))
ldd = {str(path): subprocess.check_output(["ldd", str(path)], text=True) for path in sorted(set(extensions))}

result = {
    "ep_arm": arm,
    "identity": identity,
    "torch_nccl_build": list(build_tuple),
    "nccl_runtime": runtime.value,
    "loaded_nccl_paths": mapped,
    "resolved_nccl_paths": resolved,
    "extension_ldd": ldd,
}
pathlib.Path("/opt/benchmark/image-verification.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(result, indent=2, sort_keys=True))
if build != runtime.value:
    raise SystemExit(
        f"NCCL_IDENTITY_GATE_FAILED torch build={build} runtime={runtime.value}; "
        "the exact NeMo image must be rebuilt against NCCL 2.31.2 before acceptance"
    )
print("NCCL_IDENTITY_GATE PASS")
PY
