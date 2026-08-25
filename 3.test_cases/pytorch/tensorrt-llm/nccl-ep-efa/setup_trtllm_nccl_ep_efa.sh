#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
#
# setup_trtllm_nccl_ep_efa.sh — build aws-ofi-nccl's GIN (CPU-proxy) plugin for the
# TensorRT-LLM NcclEP backend on EFA. `nccl.ep` (nccl4py) drives its network traffic through
# NCCL, and on EFA that means aws-ofi-nccl with GIN compiled in; the EFA installer's stock
# plugin does not carry it, hence this source build.
#
# Distinct from the NVSHMEM-path setup_deepep_efa.sh (vendor-synced via
# .github/workflows/deepep-vendor-sync.yml) and from the vLLM sample's
# setup_deepep_v2_efa.sh — this script builds NO DeepEP at all (TRT-LLM's EP backend is
# nccl.ep, not the deep_ep python package) and is intentionally NOT vendor-synced.
#
# Runs inside the Docker build (no GPU needed).
set -euo pipefail

# ---- pins (every one justified; no 'latest'; a bare refs/pull/N/head is a MOVING ref) ----
AWS_OFI_NCCL_REPO="${AWS_OFI_NCCL_REPO:-https://github.com/aws/aws-ofi-nccl.git}"
AWS_OFI_NCCL_SHA="${AWS_OFI_NCCL_SHA:-9c44d34476f90ddbf4a12d0ac4fc412d46bd8ab4}"  # GIN plugin, gdrdrv-2.4 v1-fallback baked
AWS_OFI_NCCL_PR="${AWS_OFI_NCCL_PR:-1351}"                                        # OFI_NCCL_GDRCOPY_FORCED_PCIE_COPY param
AWS_OFI_NCCL_PR_SHA="${AWS_OFI_NCCL_PR_SHA:-c2e773dfb2c75b765b3415f8ffd1b47e7c239a7b}"  # IMMUTABLE PR#1351 head

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
# --with-nccl-headers points at the pip nvidia-nccl-cu13 2.30.4 tree installed in Layer 4:
# GIN needs the ncclGin* device API headers that only >= 2.30.x carries.
./configure --prefix=/opt/aws-ofi-nccl --with-libfabric=/opt/amazon/efa --with-cuda=/usr/local/cuda \
  --with-nccl-headers="$(python3 -c 'import nvidia.nccl; print(nvidia.nccl.__path__[0])')/include" \
  --enable-cudart-dynamic --enable-platform-aws
make -C src -j"$(nproc)"; make -C src install
test -f /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so
[ "$(nm -D /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep -c ncclGinPlugin)" -ge 1 ]  # GIN symbol present (fail-loud)
# GIN needs gdrcopy COMPILED IN (gdrapi.h at configure time) — a gdrapi-less build carries
# this exact runtime-warn string and fails nccl_ofi_gin_init at serve. Assert absence.
[ "$(strings /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep -c 'GDRCopy support not available at compile time')" -eq 0 ]
ldconfig
cd /; rm -rf /opt/aws-ofi-nccl-src
echo "== setup_trtllm_nccl_ep_efa.sh complete: GIN plugin at /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so =="
