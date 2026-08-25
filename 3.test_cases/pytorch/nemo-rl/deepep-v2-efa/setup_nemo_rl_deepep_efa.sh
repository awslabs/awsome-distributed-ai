#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# setup_nemo_rl_deepep_efa.sh — build the two source components that carry
# DeepEP V2's MoE all-to-all onto AWS EFA for the NeMo-RL test case:
#   ofi     aws-ofi-nccl with GIN (CPU-proxy) compiled in. DeepEP V2's NCCL
#           backend drives its network traffic through NCCL's GIN device API,
#           and the EFA installer's stock plugin does not carry GIN — hence
#           this source build.
#   deepep  the deep_ep python package (upstream deepseek-ai/DeepEP, EPv2)
#           compiled against the GIN-capable NCCL under $NCCL_HOME. Run AFTER
#           the optional draft-PR patch layer so patched kernels compile in.
#
# Distinct from the NVSHMEM-path setup_deepep_efa.sh (vendor-synced via
# .github/workflows/deepep-vendor-sync.yml) — this script builds the NCCL-GIN
# V2 backend, links NO NVSHMEM, and is intentionally NOT vendor-synced.
#
# Runs inside the Docker build (no GPU needed).
set -euo pipefail

PHASE="${1:?usage: setup_nemo_rl_deepep_efa.sh {ofi|deepep}}"

# ---- pins (defaults match the Dockerfile ARGs; immutable SHAs only) --------
AWS_OFI_NCCL_REPO="${AWS_OFI_NCCL_REPO:-https://github.com/aws/aws-ofi-nccl.git}"
AWS_OFI_NCCL_SHA="${AWS_OFI_NCCL_SHA:-9c44d34476f90ddbf4a12d0ac4fc412d46bd8ab4}"  # GIN plugin, gdrdrv-2.4 v1-fallback baked
AWS_OFI_NCCL_PR="${AWS_OFI_NCCL_PR:-1351}"                                        # OFI_NCCL_GDRCOPY_FORCED_PCIE_COPY param
AWS_OFI_NCCL_PR_SHA="${AWS_OFI_NCCL_PR_SHA:-c2e773dfb2c75b765b3415f8ffd1b47e7c239a7b}"  # IMMUTABLE PR#1351 head
NCCL_HOME="${NCCL_HOME:-/opt/nccl/build}"
DEEPEP_SRC="${DEEPEP_SRC:-/opt/DeepEP}"

build_ofi() {
  echo "== aws-ofi-nccl GIN @ ${AWS_OFI_NCCL_SHA} + PR#${AWS_OFI_NCCL_PR} =="
  git clone "${AWS_OFI_NCCL_REPO}" /opt/aws-ofi-nccl-src
  cd /opt/aws-ofi-nccl-src
  git config user.email build@local; git config user.name build
  git fetch origin "${AWS_OFI_NCCL_SHA}"; git checkout "${AWS_OFI_NCCL_SHA}"
  grep -q FALLBACK_V1_FOR_GDRDRV_24 src/nccl_ofi_gdrcopy.cpp   # assert the v1-fallback is present (fail-loud)
  if [ -n "${AWS_OFI_NCCL_PR_SHA}" ]; then
    git fetch origin "${AWS_OFI_NCCL_PR_SHA}"
    git cherry-pick "${AWS_OFI_NCCL_PR_SHA}"
    grep -q GDRCOPY_FORCED_PCIE_COPY include/nccl_ofi_param.h  # assert the param landed (fail-loud)
  fi
  git rev-parse HEAD > /opt/aws-ofi-nccl.effective.sha
  ./autogen.sh
  # --with-nccl-headers: the ncclGin* device API headers only >= 2.30.x carries,
  # taken from the source-built GIN NCCL under $NCCL_HOME (Dockerfile Layer 4).
  ./configure --prefix=/opt/aws-ofi-nccl --with-libfabric=/opt/amazon/efa --with-cuda=/usr/local/cuda \
    --with-nccl-headers="${NCCL_HOME}/include" \
    --enable-cudart-dynamic --enable-platform-aws
  make -C src -j"$(nproc)"; make -C src install
  test -f /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so
  [ "$(nm -D /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep -c ncclGinPlugin)" -ge 1 ]  # GIN symbol present (fail-loud)
  # GIN needs gdrcopy COMPILED IN (gdrapi.h at configure time) — a gdrapi-less
  # build carries this exact runtime-warn string and fails GIN init on first
  # use. Assert the string is ABSENT from the built plugin.
  [ "$(strings /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep -c 'GDRCopy support not available at compile time')" -eq 0 ]
  ldconfig
  cd /; rm -rf /opt/aws-ofi-nccl-src
  echo "== ofi phase complete: GIN plugin at /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so =="
}

build_deepep() {
  echo "== DeepEP V2 (NCCL backend) from ${DEEPEP_SRC} against NCCL at ${NCCL_HOME} =="
  [ -d "${DEEPEP_SRC}" ] || { echo "ERROR: ${DEEPEP_SRC} missing — Dockerfile Layer 6 clones it" >&2; exit 1; }
  test -f "${NCCL_HOME}/include/nccl_device.h" \
    || { echo "ERROR: ${NCCL_HOME}/include/nccl_device.h missing — NCCL is not GIN-capable" >&2; exit 1; }
  # The V2 setup reads EP_NCCL_ROOT_DIR for the NCCL backend's headers/libs;
  # assert this tree actually carries the NCCL backend (EPv2) before building,
  # so a wrong-SHA checkout fails here and not with a distant include error.
  grep -q EP_NCCL_ROOT_DIR "${DEEPEP_SRC}/setup.py" \
    || { echo "ERROR: ${DEEPEP_SRC}/setup.py has no EP_NCCL_ROOT_DIR — not an EPv2 (NCCL backend) tree" >&2; exit 1; }
  cd "${DEEPEP_SRC}"
  git rev-parse HEAD > /opt/deepep.effective.sha
  export EP_NCCL_ROOT_DIR="${EP_NCCL_ROOT_DIR:-${NCCL_HOME}}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
  # --no-deps: deep_ep's metadata must not drag a second torch/NCCL into the
  # image (the pip nvidia-nccl wheels are exactly what Layer 4 removed).
  pip3 install --no-cache-dir --no-build-isolation --no-deps -v .
  # Build sandbox has no GPU, so no import smoke here (recipe/verify-image.sh
  # does that on an EFA/GPU host). Assert the compiled extension landed.
  SO_COUNT=$(python3 - <<'PY'
import glob, importlib.util, os, sys
spec = importlib.util.find_spec("deep_ep")
if spec is None or not spec.submodule_search_locations:
    sys.exit("deep_ep not installed")
root = list(spec.submodule_search_locations)[0]
print(len(glob.glob(os.path.join(os.path.dirname(root), "deep_ep*", "**", "*.so"), recursive=True)
          + glob.glob(os.path.join(root, "**", "*.so"), recursive=True)))
PY
)
  [ "${SO_COUNT}" -ge 1 ] || { echo "ERROR: no compiled deep_ep extension (.so) found after install" >&2; exit 1; }
  echo "== deepep phase complete: deep_ep built @ $(cat /opt/deepep.effective.sha) =="
}

case "${PHASE}" in
  ofi)    build_ofi ;;
  deepep) build_deepep ;;
  *) echo "FATAL: unknown phase '${PHASE}' (ofi|deepep)" >&2; exit 2 ;;
esac
