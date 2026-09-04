#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# build_deepep.sh — build DeepEP-V2 _C.so IN-POD (needs a live CUDA context, so it
# cannot run in the Docker build sandbox). Run ONCE on first pod boot.
#
# The DeepEP source + PR612 + multi-comm overlay are already staged at /opt/DeepEP by
# the Dockerfile. This compiles the CUDA extension and pip-installs it editable so
# `import deep_ep` resolves to /opt/DeepEP with ElasticBuffer present.
#
# CUDA toolchain: the image's torch is cu130, so we build with the cu13 nvcc toolchain
# (pip nvidia-cuda-nvcc-cu13) to match the torch ABI. The path is AUTO-DISCOVERED (the
# python minor version differs across base images, so we never hardcode python3.NN).
# If the cu13 nvcc cannot be obtained, we fall back to the base /usr/local/cuda nvcc.
#
# Verify after: python3 -c "import deep_ep; print(hasattr(deep_ep,'ElasticBuffer'))" -> True
set -euo pipefail

DEEPEP_DIR="${DEEPEP_DIR:-/opt/DeepEP}"
test -d "$DEEPEP_DIR" || { echo "FATAL: $DEEPEP_DIR missing (Dockerfile DeepEP layer should have cloned it)"; exit 2; }

# ---- discover the python site-packages dir (version-agnostic) ---------------
SITE="$(python3 -c 'import site;print(site.getsitepackages()[0])')"
echo "=== site-packages: $SITE ==="

# ---- FAST PATH: if the BASE /usr/local/cuda is already CUDA 13.x, use it directly -----------
# A CUDA-13 NGC base (nvcr.io/nvidia/cuda:13.0.x) ships a COHERENT toolkit (nvcc + cudart headers +
# cccl all on the same minor) that matches torch cu130. Using it avoids the entire fragile cu13-pip
# reconstruction (the nvcc/crt/nvvm/runtime/cccl/cublas version-web that fails on a cu12.9 base).
CU13=""
FAST_BASE=""
if [ -x /usr/local/cuda/bin/nvcc ] && /usr/local/cuda/bin/nvcc --version 2>/dev/null | grep -q "release 13"; then
  echo "=== FAST PATH: base /usr/local/cuda is CUDA 13.x ($(/usr/local/cuda/bin/nvcc --version|grep -i release)) — using it, no cu13 pip reconstruction ==="
  CU13=/usr/local/cuda
  FAST_BASE=1   # base is coherent CUDA-13: do NOT let find_cu13 override it with a stale pip cu13
fi

# ---- otherwise (cu12.x base): locate or pip-reconstruct a cu13 nvcc toolchain ---------------
find_cu13() {
  for d in "$SITE"/nvidia/cu13 "$SITE"/nvidia/cuda_nvcc /usr/local/lib/python3.*/dist-packages/nvidia/cu13; do
    [ -x "$d/bin/nvcc" ] && { echo "$d"; return 0; }
  done
  return 1
}
[ -z "$CU13" ] && [ -z "$FAST_BASE" ] && CU13="$(find_cu13 || true)"
if [ -z "$FAST_BASE" ] && { [ -z "$CU13" ] || [ ! -x "$CU13/bin/nvcc" ]; }; then
  # The cu13 nvcc ships as the UNSUFFIXED `nvidia-cuda-nvcc` (version 13.x); the `-cu13`
  # suffix does not exist on PyPI (it errors as a redirect stub). 13.0.88 matches torch cu130.
  # It installs nvcc + crt + nvvm under nvidia/cu13/. Match the torch CUDA minor (13.0).
  echo "=== installing cu13 nvcc toolchain (nvcc + crt + nvvm + cccl, ALL pinned 13.0.x) ==="
  # PIN every cu13 component to the SAME version: nvidia-cuda-nvcc pulls crt+nvvm as deps, but
  # unpinned they resolve to 13.3.x while nvcc is 13.0.88 -> cicc emits PTX 9.3 that the 13.0
  # ptxas rejects ("Unsupported .version 9.3; current is 9.0"). Pinning crt+nvvm to 13.0.88
  # keeps cicc and ptxas on the same PTX ISA. cccl 13.0.50 provides nv/target (no 13.0.88 cccl).
  # --force-reinstall so a pre-existing HIGHER crt/nvvm (e.g. 13.3.x pulled as an nvcc dep on an
  # earlier run) is actually DOWNGRADED to 13.0.88; pip won't downgrade an already-satisfied req otherwise.
  # ALL five cu13 components must be the SAME CUDA minor (13.0): nvcc + crt + nvvm + RUNTIME + cccl.
  # The runtime is the subtle one — its cuda_runtime_api.h sets CUDART_VERSION, and CCCL's
  # cuda_toolkit.h hard-errors ("CUDA compiler and CUDA toolkit headers are incompatible") unless
  # CUDART_VERSION matches the nvcc compiler version. An unpinned runtime resolves to 13.3 while nvcc
  # is 13.0.88 -> the error. Pin runtime==13.0.88 too. (cccl 13.0.85 is the nearest 13.0.x cccl.)
  # NOTE: no --break-system-packages — the image ships pip 22.0.2 and that flag arrived
  # in pip 23.0; passing it makes pip exit 2 ("no such option") BEFORE installing, and
  # would then mask a wrong-ABI fallback build. (Ubuntu 22.04 python3.10 is not
  # PEP-668-managed, so the flag is unnecessary here anyway.)
  # FAIL LOUD, do NOT `|| true`: a masked pip failure here (no PyPI egress, resolver conflict,
  # no disk) is indistinguishable from success — find_cu13 then yields empty and the build
  # silently falls back to the base /usr/local/cuda, i.e. the wrong-ABI build this block exists
  # to prevent. A clear "cu13 toolchain unavailable" beats an ABI error three steps later.
  # (pipefail is set, so a pip failure through the tail pipe aborts here.)
  pip install --no-cache-dir --force-reinstall \
      "nvidia-cuda-nvcc==13.0.88" "nvidia-cuda-crt==13.0.88" "nvidia-nvvm==13.0.88" \
      "nvidia-cuda-runtime==13.0.88" "nvidia-cuda-cccl==13.0.85" 2>&1 | tail -4
  CU13="$(find_cu13 || true)"
  [ -n "$CU13" ] && [ -x "$CU13/bin/nvcc" ] || { echo "FATAL: cu13 toolchain install succeeded but no nvcc found under $SITE/nvidia/cu13 — cannot build the cu130-ABI DeepEP extension (base /usr/local/cuda would be the wrong ABI)"; exit 4; }
fi

# ---- choose the build toolchain ---------------------------------------------
if [ -n "$CU13" ] && [ -x "$CU13/bin/nvcc" ]; then
  CUDA_HOME_BUILD="$CU13"
  echo "=== build CUDA_HOME = cu13 toolchain: $CUDA_HOME_BUILD ($($CU13/bin/nvcc --version | grep -i release)) ==="
elif [ -x /usr/local/cuda/bin/nvcc ]; then
  CUDA_HOME_BUILD=/usr/local/cuda
  echo "=== WARN: cu13 nvcc unavailable; falling back to base nvcc: $($CUDA_HOME_BUILD/bin/nvcc --version | grep -i release) ==="
  echo "=== (if the _C.so import later fails with an ABI error, the cu13 toolchain is required) ==="
else
  echo "FATAL: no nvcc found (neither cu13 pip toolchain nor /usr/local/cuda)"; exit 3
fi

# DeepEP's extension device-links (dlink=True), which REQUIRES ninja (distutils backend
# cannot device-link). The Dockerfile pre-installs ninja at build time (a first-boot pip
# install assumes PyPI egress from the pod — wrong on private-subnet/air-gapped clusters);
# this guard remains only for non-canonical bases.
python3 -c "import ninja" 2>/dev/null || { echo "=== installing ninja (required for dlink CUDA ext; MISSING from image — non-canonical base?) ==="; pip install --no-cache-dir ninja 2>&1 | tail -2; }

cd "$DEEPEP_DIR"
rm -rf build/temp.* deep_ep/_C*.so 2>/dev/null || true

# ---- link-time library paths + unversioned .so symlinks ----------------------
# DeepEP's final link wants `-l:libnvshmem_host.so` and `-lcudart`. The wheels ship
# versioned libs (libnvshmem_host.so.3) without the unversioned linker name, and the
# cu13 runtime ships only a static libcudart, so we point at the base dynamic libcudart.
# nvshmem lib dir: prefer the import, but FALL BACK to the known site-packages path — on some bases
# `import nvidia.nvshmem` raises (namespace-package __file__ is None) even though the lib is present.
# nvidia.nvshmem is a NAMESPACE package (no __init__.py) => its __file__ is None on modern
# wheels, which makes `os.path.dirname(nvidia.nvshmem.__file__)` raise and (with `|| true`)
# silently yield an empty NVSHMEM_LIB -> the unversioned symlink below is never created ->
# the final link dies with `cannot find -l:libnvshmem_host.so`. Resolve via find_spec's
# submodule_search_locations (works for namespace packages), then fall back to the glob.
NVSHMEM_LIB="$(python3 -c 'import importlib.util,os
s=importlib.util.find_spec("nvidia.nvshmem")
p=(s.submodule_search_locations[0] if s and s.submodule_search_locations else None)
print(os.path.join(p,"lib") if p else "")' 2>/dev/null || true)"
# `ls -d A B` with only A present prints A but exits 2; pipefail carries that through | head,
# and since this assignment is the last command in the `||` list, `set -e` would abort the
# very fallback this line exists to provide. `|| true` keeps the recovered path and the script.
[ -d "$NVSHMEM_LIB" ] || NVSHMEM_LIB="$(ls -d "$SITE"/nvidia/nvshmem/lib /usr/local/lib/python3.*/dist-packages/nvidia/nvshmem/lib 2>/dev/null | head -1 || true)"
if [ -n "$NVSHMEM_LIB" ] && [ -d "$NVSHMEM_LIB" ]; then
  [ -e "$NVSHMEM_LIB/libnvshmem_host.so" ] || ln -sf "$(ls "$NVSHMEM_LIB"/libnvshmem_host.so.* | head -1)" "$NVSHMEM_LIB/libnvshmem_host.so" 2>/dev/null || true
fi
CUDART_DIR="/usr/local/cuda/lib64"   # dynamic libcudart.so lives here (base runtime)

echo "=== nvcc build_ext (TORCH_CUDA_ARCH_LIST=${DEEPEP_ARCH_LIST:-9.0}: 9.0=H100/H200, 10.0=B200) ==="
CUDA_HOME="$CUDA_HOME_BUILD" \
PATH="$CUDA_HOME_BUILD/bin:$PATH" \
LIBRARY_PATH="${NVSHMEM_LIB:-}:${CUDART_DIR}:${LIBRARY_PATH:-}" \
LD_LIBRARY_PATH="${NVSHMEM_LIB:-}:${CUDART_DIR}:${LD_LIBRARY_PATH:-}" \
TORCH_CUDA_ARCH_LIST="${DEEPEP_ARCH_LIST:-9.0}" \
MAX_JOBS="$(nproc)" \
python3 setup.py build_ext --inplace 2>&1 | tee /tmp/deepep-build.log | tail -30   # full log at /tmp/deepep-build.log for failed-build diagnostics

echo "=== editable install so 'import deep_ep' resolves here (best-effort) ==="
# The _C.so is now in-tree, so `import deep_ep` from /opt/DeepEP works directly. The pip
# editable install can fail (pip build isolation re-triggers a sandboxed build) but is
# NOT required — fall back to a .pth that puts /opt/DeepEP on sys.path so every
# interpreter resolves it.
pip install --no-cache-dir -e . 2>&1 | tail -3 || {
  echo "(editable install failed — using a .pth fallback instead)"
  SITE="$(python3 -c 'import site;print(site.getsitepackages()[0])')"
  echo "$DEEPEP_DIR" > "$SITE/deep_ep_src.pth"
}

echo "=== verify ElasticBuffer present (EFA-viable; legacy Buffer is NVSHMEM/IBGDA = dead on EFA) ==="
# NVSHMEM host lib must be on LD_LIBRARY_PATH at import (the _C.so links it). Reuse the
# find_spec resolution (namespace-package safe) rather than the None-yielding __file__ form.
NVSHMEM_LIB="$(python3 -c 'import importlib.util,os
s=importlib.util.find_spec("nvidia.nvshmem")
p=(s.submodule_search_locations[0] if s and s.submodule_search_locations else None)
print(os.path.join(p,"lib") if p else "")' 2>/dev/null || true)"
[ -d "$NVSHMEM_LIB" ] || NVSHMEM_LIB="$(ls -d /usr/local/lib/python3.*/dist-packages/nvidia/nvshmem/lib 2>/dev/null | head -1 || true)"   # same ls-exit-2-under-pipefail guard as above
LD_LIBRARY_PATH="${NVSHMEM_LIB:-}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
  python3 -c "import deep_ep; print('deep_ep:', deep_ep.__file__); assert hasattr(deep_ep,'ElasticBuffer'), 'ElasticBuffer MISSING'; print('ElasticBuffer: OK')"
echo "=== build_deepep DONE (remember: set LD_LIBRARY_PATH to include $NVSHMEM_LIB at serve time) ==="
