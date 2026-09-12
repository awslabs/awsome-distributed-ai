#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# build_deepep.sh — build DeepEP-V2 _C.so IN-POD (needs a live CUDA context, so it
# cannot run in the Docker build sandbox). Run ONCE on first pod boot.
#
# The DeepEP source (amazon-contributing fork, SHA-pinned) is already staged at /opt/DeepEP
# by the Dockerfile. This compiles the CUDA extension in-tree and registers /opt/DeepEP on
# sys.path via a .pth so `import deep_ep` resolves there with ElasticBuffer present.
#
# CUDA toolchain: the image's torch is cu130 and the base is nvcr.io/nvidia/cuda:13.0.x-devel,
# so the base /usr/local/cuda IS the matching toolchain — asserted below. Any other base would
# be a wrong-ABI build surfacing later as an import error, so the script stops instead.
#
# Verify after: python3 -c "import deep_ep; print(hasattr(deep_ep,'ElasticBuffer'))" -> True
set -euo pipefail

DEEPEP_DIR="${DEEPEP_DIR:-/opt/DeepEP}"
test -d "$DEEPEP_DIR" || { echo "FATAL: $DEEPEP_DIR missing (Dockerfile DeepEP layer should have cloned it)"; exit 2; }

# ---- discover the python site-packages dir (version-agnostic) ---------------
SITE="$(python3 -c 'import site;print(site.getsitepackages()[0])')"
echo "=== site-packages: $SITE ==="

# ---- toolchain: the base /usr/local/cuda MUST be CUDA 13.x (the torch cu130 ABI) -------------
# The sample owns its base image (FROM nvcr.io/nvidia/cuda:13.0.x-devel, Dockerfile), whose
# toolkit is coherent (nvcc + cudart headers + cccl on the same minor) and matches torch cu130.
# Hard-assert that instead of carrying fallbacks: the old cu12->cu13 pip reconstruction and the
# base-nvcc fallback were unreachable on this base, and the fallback's firing condition was
# exactly the wrong-ABI build it claimed to prevent. This also keeps first pod boot free of any
# pip/PyPI dependency (private-subnet/air-gapped clusters).
{ [ -x /usr/local/cuda/bin/nvcc ] && /usr/local/cuda/bin/nvcc --version 2>/dev/null | grep -q "release 13"; } \
  || { echo "FATAL: /usr/local/cuda is not a CUDA 13.x toolkit — torch is cu130; build this sample's Dockerfile (FROM nvcr.io/nvidia/cuda:13.0.x-devel), do not swap the base"; exit 3; }
CUDA_HOME_BUILD=/usr/local/cuda
echo "=== build CUDA_HOME = $CUDA_HOME_BUILD ($(/usr/local/cuda/bin/nvcc --version | grep -i release)) ==="

# DeepEP's extension device-links (dlink=True), which REQUIRES ninja (distutils backend
# cannot device-link). The Dockerfile bakes ninja at image-build time: first pod boot must
# not depend on PyPI egress (private-subnet/air-gapped clusters would fail inside a compile
# step instead of a clear network error). Missing ninja = non-canonical image; stop.
python3 -c "import ninja" 2>/dev/null || { echo "FATAL: ninja missing from image (the Dockerfile bakes it — rebuild from this sample's Dockerfile rather than pip-installing at boot)"; exit 3; }

cd "$DEEPEP_DIR"
rm -rf build/temp.* deep_ep/_C*.so 2>/dev/null || true

# ---- link-time library paths -------------------------------------------------
# DeepEP's setup.py at the pinned fork head resolves the wheels' VERSIONED sonames itself
# (_find_versioned_so / get_nvshmem_host_lib_name -> `-l:libnvshmem_host.so.3`), so no
# unversioned-symlink workaround is needed — the linker only needs the lib DIRECTORIES on
# LIBRARY_PATH. The dynamic libcudart.so comes from the base toolkit.
# nvidia.nvshmem is a NAMESPACE package (no __init__.py => __file__ is None on modern wheels),
# so resolve its lib dir via find_spec's submodule_search_locations, not __file__.
NVSHMEM_LIB="$(python3 -c 'import importlib.util,os
s=importlib.util.find_spec("nvidia.nvshmem")
p=(s.submodule_search_locations[0] if s and s.submodule_search_locations else None)
print(os.path.join(p,"lib") if p else "")' 2>/dev/null || true)"
# `ls -d A B` with only A present prints A but exits 2; pipefail carries that through | head,
# and since this assignment is the last command in the `||` list, `set -e` would abort the
# very fallback this line exists to provide. `|| true` keeps the recovered path and the script.
[ -d "$NVSHMEM_LIB" ] || NVSHMEM_LIB="$(ls -d "$SITE"/nvidia/nvshmem/lib /usr/local/lib/python3.*/dist-packages/nvidia/nvshmem/lib 2>/dev/null | head -1 || true)"
# one-line assert of the condition setup.py's soname resolution depends on:
ls "${NVSHMEM_LIB:-/nonexistent}"/libnvshmem_host.so.* >/dev/null 2>&1 || { echo "FATAL: no libnvshmem_host.so.* under '${NVSHMEM_LIB:-<unresolved>}' — the nvidia-nvshmem-cu13 wheel (Dockerfile Layer 3/5b) is missing or moved"; exit 4; }
CUDART_DIR="/usr/local/cuda/lib64"   # dynamic libcudart.so lives here (base runtime)

echo "=== nvcc build_ext (TORCH_CUDA_ARCH_LIST=${DEEPEP_ARCH_LIST:-9.0}: 9.0=H100/H200, 10.0=B200) ==="
CUDA_HOME="$CUDA_HOME_BUILD" \
PATH="$CUDA_HOME_BUILD/bin:$PATH" \
LIBRARY_PATH="${NVSHMEM_LIB:+$NVSHMEM_LIB:}${CUDART_DIR}:${LIBRARY_PATH:-}" \
LD_LIBRARY_PATH="${NVSHMEM_LIB:+$NVSHMEM_LIB:}${CUDART_DIR}:${LD_LIBRARY_PATH:-}" \
TORCH_CUDA_ARCH_LIST="${DEEPEP_ARCH_LIST:-9.0}" \
MAX_JOBS="$(nproc)" \
python3 setup.py build_ext --inplace 2>&1 | tee /tmp/deepep-build.log | tail -30   # full log at /tmp/deepep-build.log for failed-build diagnostics

echo "=== register $DEEPEP_DIR on sys.path via .pth (deterministic; pip -e cannot work here) ==="
# No `pip install -e .`: DeepEP's pyproject.toml has no [build-system] table, so pip's PEP 517
# default is an ISOLATED env (setuptools+wheel, system site-packages excluded) in which
# setup.py's top-level `from torch.utils.cpp_extension import ...` cannot resolve — the attempt
# fails deterministically on every boot. And if isolation were ever satisfied, the editable
# build would re-run build_ext WITHOUT the CUDA_HOME/LIBRARY_PATH/TORCH_CUDA_ARCH_LIST prefix
# above and could overwrite the working _C.so. The .pth — the state every boot ended up in
# anyway — is written unconditionally instead.
echo "$DEEPEP_DIR" > "$SITE/deep_ep_src.pth"

echo "=== verify ElasticBuffer present (EFA-viable; legacy Buffer is NVSHMEM/IBGDA = dead on EFA) ==="
# NVSHMEM host lib must be on LD_LIBRARY_PATH at import (the _C.so links it) — reuse the
# resolution above rather than re-deriving it.
LD_LIBRARY_PATH="${NVSHMEM_LIB:+$NVSHMEM_LIB:}/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
  python3 -c "import deep_ep; print('deep_ep:', deep_ep.__file__); assert hasattr(deep_ep,'ElasticBuffer'), 'ElasticBuffer MISSING'; print('ElasticBuffer: OK')"
echo "=== build_deepep DONE (remember: set LD_LIBRARY_PATH to include $NVSHMEM_LIB at serve time) ==="
