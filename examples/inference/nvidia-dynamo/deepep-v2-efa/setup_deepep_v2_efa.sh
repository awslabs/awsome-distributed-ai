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

# ---- pins (released tag + immutable SHA; no 'latest') ----
# v1.21.1 is the released aws-ofi-nccl tag that carries the CPU-proxy GIN op-tables this
# sample uses (src/rdma/gin/nccl_ofi_gin_api.cpp exports ncclGinPlugin_v11 + _v13; only
# _v14 is EFA-GDA-specific, which we do not use), and it is the aws-ofi-nccl version the
# canonical micro-benchmarks/expert-parallelism/deepep-v2-benchmark runs on (bundled by
# EFA installer 1.50.0) — known-good in this repo. Built from source here so gdrcopy
# support is compiled in BY CONSTRUCTION (asserted below) and the plugin version stays
# pinned independently of the installer. The plugin vendors its own GIN headers
# (3rd-party/nccl/cuda/include/nccl/gin_v13.h), so its GIN interface is not coupled to
# the pip NCCL headers — which is why no --with-nccl-headers flag is needed (and why
# that flag, not being an AC_ARG_WITH this project defines, was silently ignored before).
AWS_OFI_NCCL_REPO="${AWS_OFI_NCCL_REPO:-https://github.com/aws/aws-ofi-nccl.git}"
AWS_OFI_NCCL_REF="${AWS_OFI_NCCL_REF:?pass from the Dockerfile ARG — the pin has ONE home there; a default here would be an unreachable second copy}"
# DeepEP source = the amazon-contributing fork, same as the canonical setup_deepep_gin.sh
# (deepep-v2-benchmark), which pins this fork and states "the benchmark supports no other
# source". The fork carries the in-tree successors of deepseek PR#612's EFA work — the QP
# count clamps into [_C.min_unordered_gin_qps, _C.max_unordered_gin_qps] (elastic.py) and the
# RDMA link rate is probed from sysfs (envs.py _get_sysfs_rdma_gbs) — plus the Blackwell
# st.bulk 64-bit-operand fix (e3fd4361), so no EP_EFA_MAX_QPS/EP_EFA_RDMA_GBS env exists here.
DEEPEP_REPO="${DEEPEP_REPO:-https://github.com/amazon-contributing/DeepEP.git}"
DEEPEP_SHA="${DEEPEP_SHA:?pass from the Dockerfile ARG — the pin has ONE home there; a default here would be an unreachable second copy}"

echo "== aws-ofi-nccl GIN @ ${AWS_OFI_NCCL_REF} =="
git clone --depth 1 --branch "${AWS_OFI_NCCL_REF}" "${AWS_OFI_NCCL_REPO}" /opt/aws-ofi-nccl-src
cd /opt/aws-ofi-nccl-src
git rev-parse HEAD > /opt/aws-ofi-nccl.effective.sha
./autogen.sh
# Released v1.21.1 already attempts gdr_pin_buffer_v2 with GDR_PIN_FLAG_FORCE_PCIE and falls
# back to flags=0 on failure — the forced-PCIe attempt is the default and needs no env
# override. The gdrdrv-2.4 workaround the old dev-line pin carried (a cherry-pick of the
# closed-unmerged aws-ofi-nccl#1351) is gone; gdrdrv >= 2.5 on the compute nodes is a host
# precondition instead (see README Prerequisites).
./configure --prefix=/opt/aws-ofi-nccl --with-libfabric=/opt/amazon/efa --with-cuda=/usr/local/cuda \
  --with-gdrcopy=/usr/local \
  --enable-cudart-dynamic --enable-platform-aws
make -C src -j"$(nproc)"; make -C src install
test -f /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so
[ "$(nm -D /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep -c ncclGinPlugin)" -ge 1 ]  # GIN symbol present (fail-loud)
# GIN needs gdrcopy COMPILED IN (gdrapi.h at configure time) — a gdrapi-less build carries this
# exact runtime-warn string and fails nccl_ofi_gin_init at serve with ginType==NONE. Assert absence.
[ "$(strings /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep -c 'GDRCopy support not available at compile time')" -eq 0 ]
ldconfig
cd /; rm -rf /opt/aws-ofi-nccl-src

echo "== DeepEP-V2 source @ ${DEEPEP_SHA} (amazon-contributing fork) =="
git clone "${DEEPEP_REPO}" /opt/DeepEP
cd /opt/DeepEP
git fetch origin "${DEEPEP_SHA}"; git checkout "${DEEPEP_SHA}"
# third-party/fmt is a submodule setup.py's include_dirs assumes; without it the build only
# holds while torch keeps vendoring a compatible fmt under torch/include. Same step (and
# reason) as the canonical setup_deepep_gin.sh.
git submodule update --init --recursive
git rev-parse HEAD > /opt/deepep.effective.sha
echo "== setup_deepep_v2_efa.sh complete; DeepEP _C.so builds in-pod via recipe/build_deepep.sh =="
