# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# NVIDIA NCCL EP (nccl-extensions nccl_ep) dispatch/combine over AWS EFA, NCCL-GIN CPU proxy
# (NCCL_GIN_TYPE=2). Base: the official NGC CUDA 13.0.2 devel image, pinned by digest. Every
# other input is public and pinned by version plus SHA-256 or commit (see README "Versions").
# The build steps live in setup_nccl_ep_efa.sh, one stage per layer.
#
#   DOCKER_BUILDKIT=1 docker build -f nccl-ep.Dockerfile -t <your-registry>/nccl-ep-efa:<tag> .
#
# One setup stage per RUN on purpose, so a change to a later stage reuses the earlier layers.
# hadolint global ignore=DL3059
ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6
FROM ${CUDA_IMAGE}
ARG CUDA_IMAGE

LABEL org.opencontainers.image.description="NVIDIA NCCL EP over AWS EFA (NCCL-GIN CPU proxy)"
LABEL org.opencontainers.image.licenses="MIT-0"

ARG NCCL_EXTENSIONS_COMMIT=901c4e65d6c3c8141499902c99838573de43a253
# EP_-prefixed: the base image already sets ENV NCCL_VERSION for its own (removed) NCCL.
ARG EP_NCCL_VERSION=2.32.3
ARG NCCL_WHEEL_URL=https://files.pythonhosted.org/packages/5b/29/6b277e63c92d91f9cb4d1a3a554e148983de39d54baa652bb52c798af78e/nvidia_nccl_cu13-2.32.3-py3-none-manylinux_2_27_x86_64.whl
ARG NCCL_WHEEL_SHA256=1459723080ac889d73a26edfa3e04383a7928ab31ac8f0ec43b3ea9548b04ff3
ARG EFA_INSTALLER_VERSION=1.50.0
ARG EFA_INSTALLER_SHA256=fa6dff8593d866866c13cb4640d9059835cd4efa427971f100ab40c97bef2841
ARG GDRCOPY_COMMIT=c91ad9f178e5fb729fc5b6dc62a77c3bb364d6c9
# 9.0 = H100/H200 (p5/p5en), the only architecture this recipe was measured on.
ARG NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY setup_nccl_ep_efa.sh /opt/setup_nccl_ep_efa.sh
RUN chmod 755 /opt/setup_nccl_ep_efa.sh && /opt/setup_nccl_ep_efa.sh deps
RUN GDRCOPY_COMMIT=${GDRCOPY_COMMIT} /opt/setup_nccl_ep_efa.sh gdrcopy
RUN EFA_INSTALLER_VERSION=${EFA_INSTALLER_VERSION} EFA_INSTALLER_SHA256=${EFA_INSTALLER_SHA256} \
    /opt/setup_nccl_ep_efa.sh efa
RUN EP_NCCL_VERSION=${EP_NCCL_VERSION} NCCL_WHEEL_URL=${NCCL_WHEEL_URL} NCCL_WHEEL_SHA256=${NCCL_WHEEL_SHA256} \
    /opt/setup_nccl_ep_efa.sh nccl
RUN NCCL_EXTENSIONS_COMMIT=${NCCL_EXTENSIONS_COMMIT} NVCC_GENCODE="${NVCC_GENCODE}" CUDA_IMAGE=${CUDA_IMAGE} \
    EFA_INSTALLER_VERSION=${EFA_INSTALLER_VERSION} GDRCOPY_COMMIT=${GDRCOPY_COMMIT} \
    /opt/setup_nccl_ep_efa.sh nccl-ep \
    && rm -rf /var/lib/apt/lists/*

# Paths only. The launcher passes every runtime setting to each rank with mpirun -x, because
# ranks started over ssh do not inherit the image ENV.
ENV NCCL_HOME=/opt/nccl \
    NCCL_EP_HOME=/opt/nccl-ep \
    CUDA_HOME=/usr/local/cuda \
    NCCL_NET_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so \
    NCCL_GIN_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so \
    PATH=/opt/nccl-ep/test/nccl_ep:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/nccl/lib:/opt/nccl-ep/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/opt/gdrcopy/lib:/usr/local/cuda/lib64

CMD ["/bin/bash", "-c", "cat /opt/nccl-ep/BUILD-RECORD.txt; echo 'run: kubernetes/ + scripts/run_nccl_ep_efa.sh (see README)'"]
