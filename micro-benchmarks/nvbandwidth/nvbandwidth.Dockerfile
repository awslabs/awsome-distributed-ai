# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# nvbandwidth for P6e-GB200 (arm64 / Grace), multi-node build.
# Pinned for the Grace-Blackwell platform: arm64 base, CUDA >= 12.3, sm_100/sm_103.

ARG CUDA_VERSION=12.8.0
# NOTE: arm64/sbsa base -- Grace is ARM Neoverse. Do NOT use an x86_64 base (that would
# be a B200/B300 HGX assumption; this is GB200/Grace).
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG NVBANDWIDTH_REF=main
ARG EFA_INSTALLER_VERSION=1.48.0
ARG OPENMPI_VERSION=5.0.6

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build build-essential ca-certificates \
      libboost-program-options-dev curl pkg-config && \
    rm -rf /var/lib/apt/lists/*

# --- EFA + bundled aws-ofi-nccl + libfabric (provides the OpenMPI used at runtime) ---
RUN curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz -o /tmp/efa.tar.gz && \
    cd /tmp && tar -xf efa.tar.gz && cd aws-efa-installer && \
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify && \
    rm -rf /tmp/efa.tar.gz /tmp/aws-efa-installer
ENV PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH

# --- nvbandwidth (multi-node) ---
# -DMULTINODE=1 builds the multinode_* tests that exercise the NVL72 fabric and the
# IMEX cross-instance path. CUDA architectures: 100 (GB200), 103 (GB300).
RUN git clone https://github.com/NVIDIA/nvbandwidth.git /opt/nvbandwidth && \
    cd /opt/nvbandwidth && git checkout ${NVBANDWIDTH_REF} && \
    cmake -B build -G Ninja \
      -DMULTINODE=1 \
      -DCMAKE_CUDA_ARCHITECTURES="100;103" \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j

ENV PATH=/opt/nvbandwidth/build:$PATH
WORKDIR /opt/nvbandwidth
