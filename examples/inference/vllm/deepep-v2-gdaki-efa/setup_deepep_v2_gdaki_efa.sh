#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
#
# setup_deepep_v2_gdaki_efa.sh — build aws-ofi-nccl with the GDAKI (GPU-initiated,
# kernel-posted WQE) GIN backend + stage DeepEP-V2 source for the vLLM deepep_v2 backend
# on EFA. This is the NCCL_GIN_TYPE=3 counterpart to ../deepep-v2-efa/setup_deepep_v2_efa.sh
# (which builds the NCCL_GIN_TYPE=2 CPU-proxy plugin) — same DeepEP base+PR, the delta is
# --enable-gdaki + the newer rdma-core/libfabric substrate the Dockerfile builds first.
# Distinct from the NVSHMEM-path setup_deepep_efa.sh gated by
# .github/workflows/deepep-vendor-sync.yml — intentionally NOT vendor-synced to that copy.
#
# Runs inside the Docker build, AFTER the Dockerfile has built /opt/rdma-core-gdaki
# (post-PR#1701 comp-cntr verbs) + /opt/libfabric-gdaki (post-PR#12591) and pip-installed
# the cu13 torch stack. The DeepEP _C.so itself is compiled IN-POD at first boot
# (recipe/build_deepep.sh) because it needs a live CUDA context the build sandbox lacks.
set -euo pipefail

# ---- pins (every one justified; no 'latest'; a bare refs/pull/N/head is a MOVING ref) ----
# aws-ofi-nccl is a plain SHA pin — NO local cherry-pick. The forced-PCIe path aws-ofi-nccl
# PR#1351 targeted was CLOSED by the maintainer in favour of requiring GDRCopy 2.5+ on the
# node, so this build carries no #1351 patch; GDRCopy 2.5+ is a documented NODE PRECONDITION
# instead (see README + Dockerfile Layer 3).
AWS_OFI_NCCL_REPO="${AWS_OFI_NCCL_REPO:-https://github.com/aws/aws-ofi-nccl.git}"
AWS_OFI_NCCL_SHA="${AWS_OFI_NCCL_SHA:-a3d268024576c159e97916151666c2ef20f91813}"  # master lineage: PR#1311 hw-counter tristate + 6e504db GIN seq-space fix + 80f2c78 auto-GDAKI
DEEPEP_REPO="${DEEPEP_REPO:-https://github.com/deepseek-ai/DeepEP.git}"
DEEPEP_SHA="${DEEPEP_SHA:-b306af06afd412c88e51e71802951606e40b7358}"             # measured substrate base
DEEPEP_PR="${DEEPEP_PR:-612}"                                                     # filed upstream, OPEN — EFA auto-QP cap
DEEPEP_PR_SHA="${DEEPEP_PR_SHA:-28d1f7fb173f728be51632ce0026fea23243e350}"        # IMMUTABLE PR#612 head (moving-ref trap)

echo "== aws-ofi-nccl GDAKI @ ${AWS_OFI_NCCL_SHA} (SHA pin, no local patches) =="
git clone "${AWS_OFI_NCCL_REPO}" /opt/aws-ofi-nccl-src
cd /opt/aws-ofi-nccl-src
git config user.email build@local; git config user.name build
git fetch origin "${AWS_OFI_NCCL_SHA}"; git checkout "${AWS_OFI_NCCL_SHA}"
grep -rq GDAKI_EFA_HW_COUNTER include/ src/                      # assert the PR#1311 hw-counter tristate present (fail-loud)
git rev-parse HEAD > /opt/aws-ofi-nccl-gdaki.sha
./autogen.sh
# --enable-gdaki (the GPU-initiated backend) against the newer libfabric+rdma-core the Dockerfile built.
./configure --prefix=/opt/aws-ofi-nccl-gdaki \
  --with-libfabric=/opt/libfabric-gdaki \
  --with-cuda=/usr/local/cuda \
  --with-nccl-headers="$(python3 -c 'import nvidia.nccl; print(nvidia.nccl.__path__[0])')/include" \
  --enable-gdaki \
  --enable-cudart-dynamic --enable-platform-aws
make -C src -j"$(nproc)"; make -C src install
test -f /opt/aws-ofi-nccl-gdaki/lib/libnccl-net-ofi.so
[ "$(nm -D /opt/aws-ofi-nccl-gdaki/lib/libnccl-net-ofi.so | grep -c ncclGinPlugin)" -ge 1 ]                 # GIN symbol present (fail-loud)
[ "$(strings /opt/aws-ofi-nccl-gdaki/lib/libnccl-net-ofi.so | grep -ci gdaki)" -ge 1 ]                     # GDAKI code compiled in (fail-loud)
[ "$(strings /opt/aws-ofi-nccl-gdaki/lib/libnccl-net-ofi.so | grep -c GDAKI_EFA_HW_COUNTER)" -ge 1 ]       # hw-counter tristate baked (fail-loud)
ldconfig
cd /; rm -rf /opt/aws-ofi-nccl-src

echo "== DeepEP-V2 source @ ${DEEPEP_SHA} + PR#${DEEPEP_PR} (@ ${DEEPEP_PR_SHA}, filed upstream, open) =="
git clone "${DEEPEP_REPO}" /opt/DeepEP
cd /opt/DeepEP
git config user.email build@local; git config user.name build
git fetch origin "${DEEPEP_SHA}"; git checkout "${DEEPEP_SHA}"
git rev-parse HEAD > /opt/deepep.base.sha
# PR#612 (deepseek-ai/DeepEP) — EFA auto-QP cap, filed upstream and still OPEN. Merged in by its
# IMMUTABLE head SHA (not the moving refs/pull ref). This is the ONE upstream reference this
# folder carries; there is NO local source patch. When #612 merges, re-pin DEEPEP_SHA to the
# merge commit and drop DEEPEP_PR / DEEPEP_PR_SHA (the merge becomes a plain post-merge SHA pin).
git fetch origin "refs/pull/${DEEPEP_PR}/head"
git merge --no-edit "${DEEPEP_PR_SHA}"                       # pin the IMMUTABLE PR head, not the moving ref
git rev-parse HEAD > /opt/deepep.effective.sha
test -f /opt/DeepEP/tests/elastic/test_ep.py
test -f /opt/DeepEP/csrc/elastic/buffer.hpp
echo "== setup_deepep_v2_gdaki_efa.sh complete; DeepEP _C.so builds in-pod via recipe/build_deepep.sh =="
