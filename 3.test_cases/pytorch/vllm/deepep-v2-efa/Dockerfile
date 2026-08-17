# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
#
# vLLM + DeepEP-V2 MoE all-to-all over AWS EFA (NCCL-GIN CPU-proxy). NGC-from-scratch.
# DeepEP's default NVSHMEM/IBGDA transport is dead on EFA; this uses aws-ofi-nccl's GIN
# CPU-proxy (NCCL_GIN_TYPE=2). Build logic lives in setup_deepep_v2_efa.sh (COPY'd, not
# curled) so it is in-tree + reviewable. The DeepEP _C.so is built IN-POD on first boot
# (recipe/build_deepep.sh) — it needs a live CUDA context the build sandbox lacks.
#
#   DOCKER_BUILDKIT=1 docker build --build-arg AWS_OFI_NCCL_PR_SHA=<pr1354-head> -t vllm-deepep-v2-efa:latest .
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
ARG EFA_INSTALLER_VER=1.49.0                          # pin — no 'latest' (the stack under test)
RUN apt-get update \
    && curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VER}.tar.gz | tar -xzf - -C /tmp \
    && cd /tmp/aws-efa-installer && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify \
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
    && pip3 install --no-cache-dir "nvidia-nccl-cu13==${NVIDIA_NCCL_CU13}" "nvidia-nvshmem-cu13==${NVSHMEM_CU13}"
# resolve the pip nccl onto the linker path (dynamic — never hardcode python3.NN) so it wins over any system libnccl
RUN NCCL_ROOT=$(python3 -c "import nvidia.nccl, pathlib; print(pathlib.Path(nvidia.nccl.__path__[0]))") \
    && echo "$NCCL_ROOT/lib" > /etc/ld.so.conf.d/00-pip-nccl.conf && ldconfig \
    && ldconfig -p | grep -q "libnccl.so.2 => $NCCL_ROOT/lib"   # assert 2.30.4 wins (fail-loud; the ncclTeamWorld null-deref guard)

# ---- Layer 4: aws-ofi-nccl GIN + DeepEP-V2 source (in-tree script, COPY'd not curled) ----
COPY setup_deepep_v2_efa.sh /opt/setup_deepep_v2_efa.sh
ARG AWS_OFI_NCCL_SHA=9c44d34476f90ddbf4a12d0ac4fc412d46bd8ab4
ARG AWS_OFI_NCCL_PR=1354
ARG AWS_OFI_NCCL_PR_SHA=""                             # pass the immutable head of PR#1354 at build
ARG DEEPEP_SHA=b306af06afd412c88e51e71802951606e40b7358
ARG DEEPEP_PR=612
ARG DEEPEP_PR_SHA=28d1f7fb173f728be51632ce0026fea23243e350
RUN chmod +x /opt/setup_deepep_v2_efa.sh \
    && AWS_OFI_NCCL_SHA=${AWS_OFI_NCCL_SHA} AWS_OFI_NCCL_PR=${AWS_OFI_NCCL_PR} AWS_OFI_NCCL_PR_SHA=${AWS_OFI_NCCL_PR_SHA} \
       DEEPEP_SHA=${DEEPEP_SHA} DEEPEP_PR=${DEEPEP_PR} DEEPEP_PR_SHA=${DEEPEP_PR_SHA} \
       /opt/setup_deepep_v2_efa.sh
ENV LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:${LD_LIBRARY_PATH}

# ---- Layer 5: vLLM (pinned wheel = the first commit with the deepep_v2 backend, PR#41183) ----
ARG VLLM_SHA=e2f993dc4116ba96ea10ec40b7bd2ee45b27b2b8
ARG VLLM_WHEEL_URL="https://wheels.vllm.ai/${VLLM_SHA}/vllm-0.22.1rc1.dev283%2Bge2f993dc4-cp38-abi3-manylinux_2_28_x86_64.whl"
RUN pip3 install --no-cache-dir "${VLLM_WHEEL_URL}" \
    && python3 -c "import vllm; print('vllm', vllm.__version__)"

# ---- Layer 6: the serve/build scripts (LAST — script iteration never invalidates heavy layers) ----
COPY recipe/build_deepep.sh /opt/build_deepep.sh
COPY recipe/serve.sh /opt/serve.sh
COPY recipe/run-kernel-test.sh /opt/run-kernel-test.sh
COPY recipe/benchmark_probe.py /opt/benchmark_probe.py
RUN chmod +x /opt/build_deepep.sh /opt/serve.sh /opt/run-kernel-test.sh

# The segfault-fix env (EP_REUSE_NCCL_COMM=0) + proxy-Gin contract are set by recipe/serve.sh at launch.
CMD ["/bin/bash", "-lc", "echo 'run: recipe/serve.sh {leader|worker} <ip>; build_deepep.sh runs once on first boot'; sleep infinity"]
