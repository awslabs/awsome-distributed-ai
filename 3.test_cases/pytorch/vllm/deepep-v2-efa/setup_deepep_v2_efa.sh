#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
#
# setup_deepep_v2_efa.sh — build aws-ofi-nccl (GIN CPU-proxy) + stage DeepEP-V2 source for the
# vLLM deepep_v2 backend on EFA. Distinct from the NVSHMEM-path setup_deepep_efa.sh (which pins
# DeepEP 567632d and is gated by .github/workflows/deepep-vendor-sync.yml) — this is the V2 /
# NCCL-GIN counterpart and is intentionally NOT vendor-synced to that canonical copy.
#
# Runs inside the Docker build. The DeepEP _C.so itself is compiled IN-POD at first boot
# (recipe/build_deepep.sh) because it needs a live CUDA context the build sandbox lacks.
set -euo pipefail

# ---- pins (every one justified; no 'latest'; a bare refs/pull/N/head is a MOVING ref) ----
AWS_OFI_NCCL_REPO="${AWS_OFI_NCCL_REPO:-https://github.com/aws/aws-ofi-nccl.git}"
AWS_OFI_NCCL_SHA="${AWS_OFI_NCCL_SHA:-9c44d34476f90ddbf4a12d0ac4fc412d46bd8ab4}"  # GIN plugin, gdrdrv-2.4 v1-fallback baked
AWS_OFI_NCCL_PR="${AWS_OFI_NCCL_PR:-1351}"                                       # OFI_NCCL_GDRCOPY_FORCED_PCIE_COPY param
AWS_OFI_NCCL_PR_SHA="${AWS_OFI_NCCL_PR_SHA:-c2e773dfb2c75b765b3415f8ffd1b47e7c239a7b}"  # IMMUTABLE PR#1351 head (a bare refs/pull/N/head is a moving ref)
DEEPEP_REPO="${DEEPEP_REPO:-https://github.com/deepseek-ai/DeepEP.git}"
DEEPEP_SHA="${DEEPEP_SHA:-b306af06afd412c88e51e71802951606e40b7358}"            # measured substrate base
DEEPEP_PR="${DEEPEP_PR:-612}"                                                    # EFA auto-QP cap
DEEPEP_PR_SHA="${DEEPEP_PR_SHA:-28d1f7fb173f728be51632ce0026fea23243e350}"       # IMMUTABLE PR#612 head (moving-ref trap)

echo "== aws-ofi-nccl GIN @ ${AWS_OFI_NCCL_SHA} + PR#${AWS_OFI_NCCL_PR} =="
git clone "${AWS_OFI_NCCL_REPO}" /opt/aws-ofi-nccl-src
cd /opt/aws-ofi-nccl-src
git config user.email build@local; git config user.name build
git fetch origin "${AWS_OFI_NCCL_SHA}"; git checkout "${AWS_OFI_NCCL_SHA}"
grep -q FALLBACK_V1_FOR_GDRDRV_24 src/nccl_ofi_gdrcopy.cpp   # assert the v1-fallback is present (fail-loud)
if [ -n "${AWS_OFI_NCCL_PR_SHA}" ]; then
  git fetch origin "refs/pull/${AWS_OFI_NCCL_PR}/head"
  git cherry-pick "${AWS_OFI_NCCL_PR_SHA}"
  grep -q GDRCOPY_FORCED_PCIE_COPY include/nccl_ofi_param.h  # assert the param landed (fail-loud)
fi
./autogen.sh
./configure --prefix=/opt/aws-ofi-nccl --with-libfabric=/opt/amazon/efa --with-cuda=/usr/local/cuda \
  --with-nccl-headers="$(python3 -c 'import nvidia.nccl; print(nvidia.nccl.__path__[0])')/include" \
  --enable-cudart-dynamic --enable-platform-aws
make -C src -j"$(nproc)"; make -C src install
test -f /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so
[ "$(nm -D /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep -c ncclGinPlugin)" -ge 1 ]  # GIN symbol present (fail-loud)
# GIN needs gdrcopy COMPILED IN (gdrapi.h at configure time) — a gdrapi-less build carries this
# exact runtime-warn string and fails nccl_ofi_gin_init at serve with ginType==NONE. Assert absence.
[ "$(strings /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep -c 'GDRCopy support not available at compile time')" -eq 0 ]
ldconfig
cd /; rm -rf /opt/aws-ofi-nccl-src

echo "== DeepEP-V2 source @ ${DEEPEP_SHA} + PR#${DEEPEP_PR} (@ ${DEEPEP_PR_SHA}) =="
git clone "${DEEPEP_REPO}" /opt/DeepEP
cd /opt/DeepEP
git config user.email build@local; git config user.name build
git fetch origin "${DEEPEP_SHA}"; git checkout "${DEEPEP_SHA}"
git fetch origin "refs/pull/${DEEPEP_PR}/head"
git merge --no-edit "${DEEPEP_PR_SHA}"                       # pin the IMMUTABLE PR head, not the moving ref
git rev-parse HEAD > /opt/deepep.effective.sha
test -f /opt/DeepEP/tests/elastic/test_ep.py
test -f /opt/DeepEP/csrc/elastic/buffer.hpp
echo "== setup_deepep_v2_efa.sh complete; DeepEP _C.so builds in-pod via recipe/build_deepep.sh =="
