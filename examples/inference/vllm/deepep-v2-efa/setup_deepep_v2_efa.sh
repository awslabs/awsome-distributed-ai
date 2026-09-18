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
# _C.so) is compiled IN-POD at first boot (recipe/build_deepep.sh) because it needs a live
# CUDA context the build sandbox lacks.
set -euo pipefail

# ---- pins (every one justified; no 'latest'; a bare refs/pull/N/head is a MOVING ref) ----
# DeepEP source divergence from the canonical setup_deepep_gin.sh (deepep-v2-benchmark): the canonical
# pins the amazon-contributing/DeepEP fork; this sample pins deepseek-ai/DeepEP@b306af06 + PR#612 — the
# substrate the shipped H200 (sm_90) numbers were measured on. The cost is Blackwell-only: the fork
# carries the st.bulk 64-bit-operand fix (amazon-contributing/DeepEP#3, merged 2026-08-24) that makes
# CUDA 13.0 codegen work on p6/Blackwell; this pin does not, so the manifest's DEEPEP_ARCH_LIST=10.x
# knobs are documented-not-verified (README "Known limitations"). To enable Blackwell, set
# DEEPEP_REPO=https://github.com/amazon-contributing/DeepEP.git + a fork SHA and re-verify on p6.
DEEPEP_REPO="${DEEPEP_REPO:-https://github.com/deepseek-ai/DeepEP.git}"
DEEPEP_SHA="${DEEPEP_SHA:-b306af06afd412c88e51e71802951606e40b7358}"            # measured substrate base (H200 sm_90)
DEEPEP_PR="${DEEPEP_PR:-612}"                                                    # EFA auto-QP cap
DEEPEP_PR_SHA="${DEEPEP_PR_SHA:-28d1f7fb173f728be51632ce0026fea23243e350}"       # IMMUTABLE PR#612 head (moving-ref trap)

echo "== DeepEP-V2 source @ ${DEEPEP_SHA} + PR#${DEEPEP_PR} (@ ${DEEPEP_PR_SHA}) =="
git clone "${DEEPEP_REPO}" /opt/DeepEP
cd /opt/DeepEP
git config user.email build@local; git config user.name build
git fetch origin "${DEEPEP_SHA}"; git checkout "${DEEPEP_SHA}"
git fetch origin "${DEEPEP_PR_SHA}"
git merge --no-edit "${DEEPEP_PR_SHA}"                       # pin the IMMUTABLE PR head, not the moving ref
git rev-parse HEAD > /opt/deepep.effective.sha
test -f /opt/DeepEP/tests/elastic/test_ep.py
test -f /opt/DeepEP/csrc/elastic/buffer.hpp
echo "== setup_deepep_v2_efa.sh complete; DeepEP _C.so builds in-pod via recipe/build_deepep.sh =="
