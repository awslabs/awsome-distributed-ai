# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
#
# vLLM + DeepEP-V2 MoE all-to-all over AWS EFA (NCCL-GIN CPU-proxy). NGC-from-scratch.
# DeepEP's default NVSHMEM/IBGDA transport is dead on EFA; this uses aws-ofi-nccl's GIN
# CPU-proxy (NCCL_GIN_TYPE=2). Build logic lives in setup_deepep_v2_efa.sh (COPY'd, not
# curled) so it is in-tree + reviewable. The DeepEP _C.so is built IN-POD on first boot
# (recipe/build_deepep.sh) — it needs a live CUDA context the build sandbox lacks.
#
#   setup/build-push.sh builds + pushes ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} (setup/env_vars);
#   manual equivalent: DOCKER_BUILDKIT=1 docker build -t <registry>/vllm-deepep-v2-efa:<tag> .
#
ARG CUDA_VER=13.0.0                                   # cu13 to match the torch 2.11+cu130 ABI DeepEP links
FROM nvcr.io/nvidia/cuda:${CUDA_VER}-devel-ubuntu22.04

LABEL org.opencontainers.image.description="vLLM + DeepEP-V2 MoE all-to-all over AWS EFA (NCCL-GIN CPU-proxy)"
LABEL org.opencontainers.image.licenses="MIT-0"
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# ---- Layer 1: system + build deps -------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential autoconf automake libtool pkg-config git curl wget ca-certificates \
      libnuma-dev libhwloc-dev python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Layer 2: AWS EFA (public installer; pinned, verified fresher than 1.48 per freshness rule) ----
# --disable-ngc: this NGC base ships /opt/nvidia/nvidia_entrypoint.sh, which trips the
# installer's NGC auto-detect and silently reroutes it down the libnccl-ofi-ngc path.
# We build aws-ofi-nccl from source ourselves (Layer 4), so force the normal install —
# same explicit choice as the sibling sglang/dsr1-deepep-efa and vllm/dsv3-uccl-nixl samples.
ARG EFA_INSTALLER_VER=1.49.0                          # pin — no 'latest' (the stack under test)
RUN apt-get update \
    && curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VER}.tar.gz | tar -xzf - -C /tmp \
    && cd /tmp/aws-efa-installer && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify --disable-ngc \
    && echo "${EFA_INSTALLER_VER}" > /opt/efa-installer.version \
    && rm -rf /tmp/aws-efa-installer /var/lib/apt/lists/*
ENV PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:${LD_LIBRARY_PATH:-}

# ---- Layer 3: torch cu13 stack (pinned; nccl-cu13 2.30.4 carries the GIN/LSA device symbols) ----
ARG TORCH_SPEC="torch==2.11.0"                        # DeepEP _C.so links this exact ABI
ARG TORCH_INDEX="https://download.pytorch.org/whl/cu130"
ARG NVIDIA_NCCL_CU13="2.30.4"                         # 2.28.x lacks the GIN/LSA symbols the plugin needs
ARG NVSHMEM_CU13="3.6.5"
RUN pip3 install --no-cache-dir --index-url "${TORCH_INDEX}" "${TORCH_SPEC}" \
    && pip3 install --no-cache-dir "nvidia-nccl-cu13==${NVIDIA_NCCL_CU13}" "nvidia-nvshmem-cu13==${NVSHMEM_CU13}" ninja
# ninja is baked here EXPLICITLY (build_deepep.sh device-links the DeepEP extension, which
# requires it): first pod boot must not depend on PyPI egress — private-subnet/air-gapped
# clusters would otherwise fail inside a compile step instead of a clear network error.
# resolve the pip nccl onto the linker path (dynamic — never hardcode python3.NN) so it wins over any system libnccl
RUN NCCL_ROOT=$(python3 -c "import nvidia.nccl, pathlib; print(pathlib.Path(nvidia.nccl.__path__[0]))") \
    && echo "$NCCL_ROOT/lib" > /etc/ld.so.conf.d/00-pip-nccl.conf && ldconfig \
    && ldconfig -p | grep "libnccl.so.2 " | head -1 | grep -q "$NCCL_ROOT/lib"   # assert the pinned NCCL resolves FIRST (2.28.x lacks the GIN/LSA symbols the plugin needs)

# ---- Layer 3b: gdrcopy userspace (PUBLIC: github.com/NVIDIA/gdrcopy) ----
# aws-ofi-nccl's GIN path REQUIRES gdrapi.h at configure time — without it the plugin
# compiles with "GDRCopy support not available", nccl_ofi_gin_init fails at serve, and
# DeepEP's ElasticBuffer asserts ginType==NCCL_GIN_TYPE_NONE. Userspace 2.5.2 on a
# gdrdrv-2.4 host is exactly the pairing OFI_NCCL_GDRCOPY_FORCED_PCIE_COPY=1 (#1351) fixes.
ARG GDRCOPY_VER=v2.5.2
RUN git clone --depth 1 --branch ${GDRCOPY_VER} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy && make prefix=/usr/local lib lib_install && ldconfig \
    && rm -rf /tmp/gdrcopy

# ---- Layer 4: aws-ofi-nccl GIN + DeepEP-V2 source (in-tree script, COPY'd not curled) ----
COPY setup_deepep_v2_efa.sh /opt/setup_deepep_v2_efa.sh
ARG AWS_OFI_NCCL_SHA=9c44d34476f90ddbf4a12d0ac4fc412d46bd8ab4
ARG AWS_OFI_NCCL_PR=1351
ARG AWS_OFI_NCCL_PR_SHA=c2e773dfb2c75b765b3415f8ffd1b47e7c239a7b  # IMMUTABLE PR#1351 head (FORCED_PCIE param; a bare refs/pull/N/head is a moving ref)
ARG DEEPEP_SHA=b306af06afd412c88e51e71802951606e40b7358
ARG DEEPEP_PR=612
ARG DEEPEP_PR_SHA=28d1f7fb173f728be51632ce0026fea23243e350
RUN chmod +x /opt/setup_deepep_v2_efa.sh \
    && AWS_OFI_NCCL_SHA=${AWS_OFI_NCCL_SHA} AWS_OFI_NCCL_PR=${AWS_OFI_NCCL_PR} AWS_OFI_NCCL_PR_SHA=${AWS_OFI_NCCL_PR_SHA} \
       DEEPEP_SHA=${DEEPEP_SHA} DEEPEP_PR=${DEEPEP_PR} DEEPEP_PR_SHA=${DEEPEP_PR_SHA} \
       /opt/setup_deepep_v2_efa.sh
ENV LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:${LD_LIBRARY_PATH}

# ---- Layer 5: vLLM (pinned wheel = the first commit with the deepep_v2 backend, PR#41183) ----
# The wheel FILENAME is derived from the per-commit index at build time, so bumping
# VLLM_SHA ALONE is sufficient (hardcoding the filename would embed the SHA twice and a
# lone bump 404s).
# NOTE: wheels.vllm.ai nightly wheels are garbage-collected upstream, so this pin has a
# shelf life independent of this repo; the fail-loud check below names that cause.
# Pin is now on vLLM main at the merge commit of #52632 (merged 2026-08-20), the
# empty-ExpertTokensMetadata guard that unblocks default (CUDA-graph) compilation; this
# commit also carries #47785 (shared-experts -1 sentinel fix, merged earlier). This is
# the "bump the pin past #52632" edit the README describes. Was e2f993dc (PR#41183's
# first deepep_v2 backend commit); wheel for the new SHA verified indexed 2026-08-25.
ARG VLLM_SHA=14617c2b6c1257ac0d6c7b5e05b195ca30013827
RUN WHEEL_HREF=$(curl -fsSL "https://wheels.vllm.ai/${VLLM_SHA}/vllm/" | grep -oE 'href="[^"]*manylinux[^"]*x86_64\.whl"' | head -1 | cut -d'"' -f2) \
    && { [ -n "$WHEEL_HREF" ] || { echo "FATAL: no x86_64 vllm wheel indexed for ${VLLM_SHA} (bad SHA, or nightly wheel garbage-collected upstream)"; exit 1; }; } \
    && pip3 install --no-cache-dir "https://wheels.vllm.ai/${VLLM_SHA}/$(basename "$WHEEL_HREF")" \
    && python3 -c "import vllm; print('vllm', vllm.__version__)"

# ---- Layer 5b: re-pin the ABI-critical cu13 libs AFTER the last pip layer (order matters) ----
# The vLLM wheel's dependency resolution drags nvidia-nccl-cu13 back down to torch's 2.28.9
# (and nvshmem to 3.4.5) — silently, into the SAME directory, so the Layer-3 linker-precedence
# assert stays green while the GIN/LSA device symbols vanish and DeepEP's nccl.cu no longer
# compiles in-pod. Any pip layer added later must keep this re-pin LAST.
RUN pip3 install --no-cache-dir --no-deps --force-reinstall \
      "nvidia-nccl-cu13==${NVIDIA_NCCL_CU13}" "nvidia-nvshmem-cu13==${NVSHMEM_CU13}" \
    && NCCL_ROOT=$(python3 -c "import nvidia.nccl, pathlib; print(pathlib.Path(nvidia.nccl.__path__[0]))") \
    && grep -rq ncclGetLsaDevicePointer "$NCCL_ROOT/include/" \
    && [ "$(nm -D "$NCCL_ROOT/lib/libnccl.so.2" | grep -c ncclGetLsaDevicePointer)" -ge 1 ]  # GIN/LSA symbols present (2.28.x lacks them)

# ---- Layer 6: the serve/build scripts (LAST — script iteration never invalidates heavy layers) ----
COPY recipe/build_deepep.sh /opt/build_deepep.sh
COPY recipe/serve.sh /opt/serve.sh
COPY recipe/run-kernel-test.sh /opt/run-kernel-test.sh
COPY recipe/benchmark_probe.py /opt/benchmark_probe.py
COPY recipe/benchmark.sh /opt/benchmark.sh
RUN chmod +x /opt/build_deepep.sh /opt/serve.sh /opt/run-kernel-test.sh /opt/benchmark.sh

# The segfault-fix env (EP_REUSE_NCCL_COMM=0) + proxy-Gin contract are set by recipe/serve.sh at launch.
CMD ["/bin/bash", "-lc", "echo 'run: recipe/serve.sh {leader|worker} <ip>; build_deepep.sh runs once on first boot'; sleep infinity"]
