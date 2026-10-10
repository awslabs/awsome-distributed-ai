#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
#
# setup_deepep_v2_efa.sh — stage DeepEP-V2 source (pinned) for the vLLM deepep_v2 backend on
# EFA. Distinct from the NVSHMEM-path setup_deepep_efa.sh (which pins DeepEP 567632d and is
# gated by .github/workflows/deepep-vendor-sync.yml) — this is the V2 / NCCL-GIN counterpart
# and is intentionally NOT vendor-synced to that canonical copy.
#
# The aws-ofi-nccl GIN plugin is NOT built here: EFA installer >= 1.50.0 bundles aws-ofi-nccl
# 1.21.1 with GA GIN support (Dockerfile Layer 2 installs it and gates on ncclGinPlugin_v14 —
# the same migration the canonical deepep-v2-benchmark made in upstream #1239).
#
# Runs inside the Docker build. This script only STAGES source — the one DeepEP BUILD (the
# _C.so) is compiled IN-POD at first boot (recipe/build_deepep.sh). That split is a design
# choice (the arch list follows the node via DEEPEP_ARCH_LIST), not a sandbox limitation:
# the canonical setup_deepep_gin.sh builds DeepEP inside `docker build`.
set -euo pipefail

# ---- pin (immutable SHA; no 'latest') ----
# DeepEP source = the amazon-contributing/DeepEP fork: the tree AWS points to for DeepEP-V2 on
# EFA, and the fork the repo's canonical V2/GIN provisioner builds from its floating main
# (deepep-v2-benchmark's setup_deepep_gin.sh: "the benchmark supports no other source"). The sibling
# vllm/deepep-v2-gdaki-efa and nvidia-dynamo/deepep-v2-efa samples pin the same SHA. The fork
# carries the in-tree successors of deepseek PR#612's EFA work -- the QP count clamps into
# [_C.min_unordered_gin_qps, _C.max_unordered_gin_qps] (deep_ep/buffers/elastic.py) and the RDMA
# link rate is probed from sysfs (deep_ep/utils/envs.py _get_sysfs_rdma_gbs) -- plus the Blackwell
# st.bulk 64-bit-operand fix (e3fd4361). So this is a plain clone at one SHA: no PR merge, no local
# patch, and no EP_EFA_MAX_QPS / EP_EFA_RDMA_GBS env anywhere in the sample.
DEEPEP_REPO="${DEEPEP_REPO:-https://github.com/amazon-contributing/DeepEP.git}"
DEEPEP_SHA="${DEEPEP_SHA:?pass from the Dockerfile ARG -- the pin has ONE home there; a default here would be an unreachable second copy}"

echo "== DeepEP-V2 source @ ${DEEPEP_SHA} (amazon-contributing fork) =="
git clone "${DEEPEP_REPO}" /opt/DeepEP
cd /opt/DeepEP
git fetch origin "${DEEPEP_SHA}"; git checkout "${DEEPEP_SHA}"
# third-party/fmt is a submodule setup.py's include_dirs assumes; without it the in-pod build
# only holds while torch keeps vendoring a compatible fmt under torch/include. Same step (and
# reason) as the canonical setup_deepep_gin.sh.
git submodule update --init --recursive
git rev-parse HEAD > /opt/deepep.effective.sha
test -f /opt/DeepEP/tests/elastic/test_ep.py
test -f /opt/DeepEP/csrc/elastic/buffer.hpp
# fork discriminator: stock deepseek-ai/DeepEP has no sysfs link-rate probe, so a DEEPEP_REPO /
# DEEPEP_SHA that silently points at the wrong tree fails the build here, not the first serve.
grep -q "_get_sysfs_rdma_gbs" /opt/DeepEP/deep_ep/utils/envs.py
echo "== setup_deepep_v2_efa.sh complete; DeepEP _C.so builds in-pod via recipe/build_deepep.sh =="
