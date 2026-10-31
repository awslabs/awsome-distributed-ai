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

# ---- pin (released tag; no 'latest') ----
# v1.21.1 is the released tag that carries the CPU-proxy GIN op-tables this sample uses
# (src/rdma/gin/nccl_ofi_gin_api.cpp exports ncclGinPlugin_v11 + _v13; only _v14 is
# EFA-GDA-specific, which we do not use). It is the same tag the sibling
# micro-benchmarks/expert-parallelism/deepep-v2-benchmark/deepep.Dockerfile builds from
# source, so it is known-good in this repo. The plugin vendors its own GIN headers
# (3rd-party/nccl/cuda/include/nccl/gin_v13.h), so its GIN interface is not coupled to the
# pip NCCL headers — which is why no --with-nccl-headers flag is needed (and why that flag,
# not being an AC_ARG_WITH this project defines, was silently ignored before).
AWS_OFI_NCCL_REPO="${AWS_OFI_NCCL_REPO:-https://github.com/aws/aws-ofi-nccl.git}"
AWS_OFI_NCCL_REF="${AWS_OFI_NCCL_REF:-v1.21.1}"

echo "== aws-ofi-nccl GIN @ ${AWS_OFI_NCCL_REF} =="
git clone --depth 1 --branch "${AWS_OFI_NCCL_REF}" "${AWS_OFI_NCCL_REPO}" /opt/aws-ofi-nccl-src
cd /opt/aws-ofi-nccl-src
git rev-parse HEAD > /opt/aws-ofi-nccl.effective.sha
./autogen.sh
# Released v1.21.1 already attempts gdr_pin_buffer_v2 with GDR_PIN_FLAG_FORCE_PCIE and falls
# back to flags=0 on failure — so the forced-PCIe attempt is the default and needs no env
# override. The gdrdrv-2.4 kernel-module fallback that the old dev-line pin carried is a host
# precondition instead: gdrdrv >= 2.5 on the compute nodes (see README Prerequisites).
./configure --prefix=/opt/aws-ofi-nccl --with-libfabric=/opt/amazon/efa --with-cuda=/usr/local/cuda \
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
