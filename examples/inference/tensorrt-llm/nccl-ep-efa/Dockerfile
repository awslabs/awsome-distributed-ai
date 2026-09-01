# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
#
# TensorRT-LLM + NcclEP MoE all-to-all over AWS EFA (NCCL-GIN CPU-proxy).
# TRT-LLM's NcclEP backend rides `nccl.ep` (nccl4py), which needs NCCL >= 2.30.4 and a
# GIN-capable network plugin — neither of which the NGC TRT-LLM container ships. This image
# adds, from PUBLIC sources only: EFA userspace, gdrcopy, the pinned NCCL + nccl4py pair,
# and aws-ofi-nccl's GIN plugin (built by setup_trtllm_nccl_ep_efa.sh — COPY'd, not curled,
# so it is in-tree + reviewable).
#
#   setup/build-push.sh builds + pushes ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} (setup/env_vars);
#   manual equivalent: DOCKER_BUILDKIT=1 docker build -t <registry>/trtllm-nccl-ep-efa:<tag> .
#
# Base pin rationale (GA-over-prerelease exception, documented): GA v1.2.1 does NOT contain
# the NcclEP backend at all (tensorrt_llm/_torch/modules/fused_moe/nccl_ep_utils.py is absent
# at that tag), so a 1.3.0rc pin is REQUIRED, not a preference. 1.3.0rc24 is the first NGC
# release container whose factory ships the NCCL_EP arm natively (TRTLLM_FORCE_COMM_METHOD);
# re-test and re-pin when a GA that carries the backend appears.
ARG TRTLLM_BASE=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc24
FROM ${TRTLLM_BASE}
ARG TRTLLM_BASE   # re-declare: pre-FROM ARGs go out of scope after FROM (used in Layer 6 diagnostics)

LABEL org.opencontainers.image.description="TensorRT-LLM + NcclEP MoE all-to-all over AWS EFA (NCCL-GIN CPU-proxy)"
LABEL org.opencontainers.image.licenses="MIT-0"
LABEL org.opencontainers.image.source="https://github.com/awslabs/awsome-distributed-ai"
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]
USER root

# ---- Layer 1: system + build deps -------------------------------------------
# libevent-{core,pthreads} are prrte-aws deps the EFA installer needs; do NOT purge
# /var/lib/apt/lists here — the installer below runs its own apt-get install and fails
# with "held broken packages" against empty lists. Purged at the end of Layer 2 instead.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential autoconf automake libtool pkg-config git curl wget ca-certificates \
      libnuma-dev libhwloc-dev libudev-dev \
      libevent-core-2.1-7t64 libevent-pthreads-2.1-7t64 \
      pciutils environment-modules tcl udev dmidecode ethtool iproute2 kmod

# ---- Layer 2: AWS EFA (public installer) ----
# 1.50.0 is the current EFA userspace and what the sibling deepep-v2-benchmark builds; its
# tarball resolves (verified) and it ships aws-ofi-nccl 1.21.1 in-box, lining this sample up
# with the rest of the repo. BUILD-PIN only: the 2026-08-07 correctness E2E (real
# trtllm-serve HTTP-200-correct + 16-rank cross-node dispatch/combine, IMA=0, efa-direct on
# every rank) ran on 1.48.0 — this assembly is not yet cluster-re-measured on 1.50.0, which
# is exactly what recipe/verify-image.sh + run-kernel-test.sh exist to re-verify on-cluster.
# --disable-ngc: the NGC base trips the installer's NGC auto-detect, which would silently
# reroute to the libnccl-ofi-ngc path — we build aws-ofi-nccl from source ourselves
# (Layer 5), same explicit choice as the sibling vllm/dsv3-uccl-nixl sample (which also
# passes --disable-ngc alone). Do NOT add --disable-build-ngc: that flag existed in
# aws-efa-installer 1.48.0 but was REMOVED in 1.49.0+ (the installer no longer auto-detects
# the build-ngc use case, so the flag became unnecessary and getopt now rejects it — an
# unknown long-opt makes efa_installer.sh print usage and exit 1, failing this layer). The
# 2026-08-07 E2E ran on 1.48.0 where both flags were valid; the 1.50.0 bump made the second
# one a build error, which is why it is dropped here.
ARG EFA_INSTALLER_VER=1.50.0
RUN apt-get update \
    && curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VER}.tar.gz | tar -xzf - -C /tmp \
    && cd /tmp/aws-efa-installer \
    && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify --disable-ngc \
    && echo "${EFA_INSTALLER_VER}" > /opt/efa-installer.version \
    && rm -rf /tmp/aws-efa-installer /var/lib/apt/lists/*
# Add EFA userspace (libfabric + efa provider) to PATH/LD, but deliberately NOT the EFA
# installer's bundled OpenMPI (/opt/amazon/openmpi). The NGC TRT-LLM base ships HPC-X OpenMPI
# (ldconfig default /opt/hpcx/ompi) and tensorrt_llm's MPI_Init resolves libopen-pal through it.
# The EFA installer's OpenMPI 4.1.7 ships a libopen-pal.so.40 that does NOT export
# opal_libevent2022_event_assign; prepending /opt/amazon/openmpi/lib here shadows the HPC-X
# libopen-pal under HPC-X's own mca_ess_hnp.so plugin → undefined symbol → MPI_Init aborts on
# `import tensorrt_llm`. Neither recipe uses EFA's mpirun (serve.sh is single-node; the probe
# uses torchrun), so the EFA OpenMPI is unnecessary and its lib on LD is actively harmful.
ENV PATH=/opt/amazon/efa/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}

# ---- Layer 3: gdrcopy userspace (PUBLIC: github.com/NVIDIA/gdrcopy) ----
# aws-ofi-nccl's GIN path REQUIRES gdrapi.h at configure time — without it the plugin
# compiles with "GDRCopy support not available", nccl_ofi_gin_init fails at serve time,
# and NcclEP's group creation dies. gdrcopy v2.5.2 == commit c91ad9f: pin the commit, not
# the tag (a bare tag is a moving ref upstream can re-point).
ARG GDRCOPY_SHA=c91ad9f178e5fb729fc5b6dc62a77c3bb364d6c9
RUN git clone https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy && git fetch origin ${GDRCOPY_SHA} && git checkout ${GDRCOPY_SHA} \
    && make prefix=/usr/local lib lib_install && ldconfig \
    && rm -rf /tmp/gdrcopy

# ---- Layer 4: NCCL 2.30.4 (over the container's baked NCCL) + nccl4py ----
# THE crux: tensorrt_llm's is_nccl_ep_installed() gates on libnccl >= 2.30.4, while the NGC
# TRT-LLM container bakes an older nvidia-nccl-cu13 — so NcclEP is dead on arrival as
# shipped. --no-deps so pip does not drag torch/TRT-LLM's pinned dependency graph backwards;
# we are deliberately overriding exactly one pin. nccl4py 0.3.1 ships the `nccl.ep` python
# package + libnccl_ep.so 0.1.0 (the version whose HT-kernel int64/int32 ABI detail the README
# documents). This nccl4py pin is REQUIRED, not merely current: nccl4py 0.4.1 ships no `nccl.ep`
# package at all (only nccl.bindings + nccl.core), so a bump to latest breaks `import nccl.ep`.
# Why 2.30.4 and not the newer 2.31.2: 2.31.2 also clears the >= 2.30.4 floor, but libnccl_ep 0.1.0
# (shipped by the nccl4py pin above) is built against the NCCL 2.30 device-side API that the GIN
# CPU-proxy path calls into. 2.30.4 is the version the 2026-08-07 correctness E2E actually ran.
# Two single-variable 2.31.2 trials on cgk p5en (--build-arg NVIDIA_NCCL_CU13=2.31.2) established:
#   (+) LIBRARY ABI is clean on 2.31.2 — is_nccl_ep_installed() passes and the CommunicationFactory
#       selects NcclEP on all 16 ranks (no quiet symbol/binding break). MEASURED.
#   (=) DEVICE dispatch/combine round-trip: NOT YET measured on either arm. First trial's GIN init
#       hard-failed because cgk's host gdrdrv is 2.4 (< the sample's GDRCopy-2.5 prereq); a second
#       trial (2026-09-01) with a throwaway forced_pcie_copy instrument patch DID init GIN on
#       gdrdrv-2.4 and both arms reached NcclEP selection, but crashed one line before the first
#       dispatch on a latent probe-diagnostic bug (recipe/probe_nccl_ep.py referenced a
#       NcclEpContext._ep_algorithm attr absent on rc24 — now getattr-guarded). See VERDICT.md.
# So the 2.31.2 device-side ABI is NOT yet certified (nor refuted) at the kernel level. Held at
# 2.30.4 as the measured-matching floor, not an upper bound — to bump it, re-run verify-image.sh +
# run-kernel-test.sh (probe fix now landed) on a gdrdrv>=2.5 host (or with the instrument patch) so
# GIN inits and the dispatch/combine round-trip + oracle actually execute on both arms.
ARG NVIDIA_NCCL_CU13=2.30.4
ARG NCCL4PY_VER=0.3.1
RUN pip3 install --no-cache-dir --no-deps --force-reinstall "nvidia-nccl-cu13==${NVIDIA_NCCL_CU13}" \
    && pip3 install --no-cache-dir "nccl4py==${NCCL4PY_VER}" \
    && NCCL_ROOT=$(python3 -c "import nvidia.nccl, pathlib; print(pathlib.Path(nvidia.nccl.__path__[0]))") \
    && ln -sf "$NCCL_ROOT/lib/libnccl.so.2" /usr/local/lib/libnccl.so.2 \
    && ln -sf "$NCCL_ROOT/lib/libnccl.so.2" /usr/local/lib/libnccl.so \
    && echo "$NCCL_ROOT/lib" > /etc/ld.so.conf.d/00-pip-nccl.conf && ldconfig \
    && [ "$(strings "$NCCL_ROOT/lib/libnccl.so.2" | grep -c "NCCL version ${NVIDIA_NCCL_CU13}")" -ge 1 ]
# The version assert uses the draining count form, not `grep -q` (SIGPIPE-141 flake under
# pipefail). Import checks (nccl.ep, is_nccl_ep_installed) live in recipe/verify-image.sh,
# which runs with GPUs — the build sandbox has none.

# ---- Layer 5: aws-ofi-nccl GIN plugin (in-tree script, COPY'd not curled) ----
# Built from the released tag v1.21.1 (same tag the sibling deepep-v2-benchmark builds): it
# exports the CPU-proxy GIN op-tables (ncclGinPlugin_v11/_v13) NCCL_GIN_TYPE=2 uses, and its
# forced-PCIe-with-fallback gdrcopy path is the released default — so no closed-PR cherry-pick
# and no OFI_NCCL_GDRCOPY_FORCED_PCIE_COPY override are needed. gdrdrv >= 2.5 is a host
# precondition (README Prerequisites) rather than a private plugin fork.
COPY setup_trtllm_nccl_ep_efa.sh /opt/setup_trtllm_nccl_ep_efa.sh
ARG AWS_OFI_NCCL_REF=v1.21.1
RUN chmod +x /opt/setup_trtllm_nccl_ep_efa.sh \
    && AWS_OFI_NCCL_REF=${AWS_OFI_NCCL_REF} /opt/setup_trtllm_nccl_ep_efa.sh
ENV LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:${LD_LIBRARY_PATH}
# NCCL_GIN_PLUGIN pairs with NCCL_NET_PLUGIN — one .so supplies both the net and GIN tables;
# set both as ENV (matching the sibling deepep-v2-benchmark) so the image is correct for
# anyone running it outside the launchers. NVIDIA_GDRCOPY=enabled matches the two sibling
# GIN images and the non-privileged/device-plugin path the manifest header offers.
ENV NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so \
    NCCL_GIN_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so \
    NVIDIA_GDRCOPY=enabled

# ---- Layer 6 (OPT-IN, default OFF): the HIGH_THROUGHPUT+FLAT selectability patch ----
# Upstream's NcclEP hardcodes LOW_LATENCY + RANK_MAJOR. On THIS substrate (NCCL 2.30.4 +
# nccl_ep 0.1.0 + GIN CPU-proxy) the upstream default runs clean — measured both in a real
# serve and at 16 ranks cross-node — so the UNPATCHED image is the baseline and this layer
# defaults OFF. NVIDIA/TensorRT-LLM PR #17715 (open) makes algorithm/layout selectable via
# TRTLLM_NCCL_EP_ALGO / TRTLLM_NCCL_EP_LAYOUT; build with APPLY_HT_FLAT_PATCH=1 to bake the
# PR's three change commits (pinned at their immutable SHAs) into the container's site-packages.
# Fail-loud: if a hunk no longer applies against this base tag, the BUILD fails — do not
# ship an image whose patch state is ambiguous. Retire this layer when #17715 merges.
ARG APPLY_HT_FLAT_PATCH=0
# The three single-parent commits of PR#17715 (NOT the branch-head merge commit
# e14a6f64 — GitHub's .patch endpoint 403s for a merge, so git format-patch emits nothing
# and the layer would die on it every time).
ARG HT_FLAT_PATCH_SHAS="6035d66353142d70ff41041b13dda1b3e788371a 2e3de8a3ccbb6bb1f78cd1504852045db989abfb 4e7cba789f4d928babc6199e895961f54ee30e11"
RUN set -euo pipefail; \
    if [ "${APPLY_HT_FLAT_PATCH}" = "1" ]; then \
      SP_PARENT=$(python3 -c "import tensorrt_llm, pathlib; print(pathlib.Path(tensorrt_llm.__file__).parent.parent)"); \
      cd "$SP_PARENT"; \
      for sha in ${HT_FLAT_PATCH_SHAS}; do \
           echo "== applying NVIDIA/TensorRT-LLM PR#17715 commit ${sha} =="; \
           curl -fsSL "https://github.com/NVIDIA/TensorRT-LLM/commit/${sha}.patch" > /tmp/${sha}.patch; \
           [ -s /tmp/${sha}.patch ] || { echo "FATAL: PR#17715 ${sha} yielded an empty patch (a merge commit has no format-patch output) — pin a single-parent commit"; exit 1; }; \
           git apply --include='tensorrt_llm/*' --check /tmp/${sha}.patch \
             || { echo "FATAL: PR#17715 ${sha} does not apply on ${TRTLLM_BASE} — base moved under the patch"; exit 1; }; \
           git apply --include='tensorrt_llm/*' /tmp/${sha}.patch; \
           rm -f /tmp/${sha}.patch; \
      done; \
      find "$SP_PARENT/tensorrt_llm/_torch/modules/fused_moe" -name '__pycache__' -prune -exec rm -rf {} + || true; \
      touch /opt/.ht-flat-patch-applied; \
    else echo "HT/FLAT patch layer skipped (APPLY_HT_FLAT_PATCH=0 — upstream LL/RANK_MAJOR default)"; fi

# ---- Layer 7: the recipe scripts (LAST — script iteration never invalidates heavy layers) ----
COPY recipe/serve.sh /opt/serve.sh
COPY recipe/run-kernel-test.sh /opt/run-kernel-test.sh
COPY recipe/probe_nccl_ep.py /opt/probe_nccl_ep.py
COPY recipe/benchmark_probe.py /opt/benchmark_probe.py
COPY recipe/benchmark.sh /opt/benchmark.sh
RUN chmod 755 /opt/serve.sh /opt/run-kernel-test.sh /opt/benchmark.sh

CMD ["/bin/bash", "-lc", "echo 'run: /opt/serve.sh (single-node EP serve) | /opt/run-kernel-test.sh {leader|worker} <ip> (cross-node NcclEP probe)'; sleep infinity"]
