# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# NVIDIA NCCL EP (nccl-extensions nccl_ep) over AWS EFA with NCCL's GIN CPU proxy (NCCL_GIN_TYPE=2).
#
#   docker build -f nccl-ep.Dockerfile -t <your-registry>/nccl-ep-efa:<tag> .
ARG CUDA_VERSION=13.0.2
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

# NCCL from the NVIDIA CUDA apt repository, which the base image already configures.
# GIN on EFA needs NCCL >= 2.31.2 and aws-ofi-nccl >= 1.21.0.
ARG NCCL_VERSION=2.32.3-1+cuda13.4
# Installs libfabric, rdma-core, aws-ofi-nccl and Open MPI.
ARG EFA_INSTALLER_VERSION=1.50.0
# aws-ofi-nccl's GIN initialization loads libgdrapi.so, in proxy mode too.
ARG GDRCOPY_VERSION=v2.5.2
# NCCL EP 0.2.0 is not in a release yet, so pin the commit.
ARG NCCL_EXTENSIONS_COMMIT=901c4e65d6c3c8141499902c99838573de43a253
# 9.0 = H100/H200 (p5, p5en), 10.0 = B200 (p6-b200), 10.3 = B300 (p6-b300). On a GPU whose
# architecture is missing here, the kernels return zeroed buffers without an error; ep_test catches it.
ARG NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_103,code=sm_103"

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Replaces the base image's held NCCL with the one NCCL EP is built against.
RUN apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
        git curl openssh-client openssh-server \
        libnccl2=${NCCL_VERSION} libnccl-dev=${NCCL_VERSION}
# The base image's ENV NCCL_VERSION still names its own NCCL; record the installed version.
ENV NCCL_VERSION=${NCCL_VERSION}

# The MPI Operator runs sshd in each worker, and the launcher reaches the workers over ssh.
RUN mkdir -p /var/run/sshd \
    && sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config \
    && echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config \
    && sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

RUN curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz | tar -xz -C /tmp \
    && cd /tmp/aws-efa-installer && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf /tmp/aws-efa-installer /var/lib/apt/lists/*

RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && make -C /tmp/gdrcopy prefix=/opt/gdrcopy lib lib_install \
    && echo /opt/gdrcopy/lib > /etc/ld.so.conf.d/gdrcopy.conf && ldconfig && rm -rf /tmp/gdrcopy

RUN git clone https://github.com/NVIDIA/nccl-extensions.git /opt/nccl-extensions \
    && git -C /opt/nccl-extensions checkout ${NCCL_EXTENSIONS_COMMIT} \
    && make -C /opt/nccl-extensions/nccl_ep -j NCCL_HOME=/usr MPI=1 MPI_HOME=/opt/amazon/openmpi \
        NVCC_GENCODE="${NVCC_GENCODE}" BUILDDIR=/opt/nccl-ep

ENV NCCL_EP_HOME=/opt/nccl-ep \
    PATH=/opt/nccl-ep/test/nccl_ep:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/nccl-ep/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/opt/gdrcopy/lib:${LD_LIBRARY_PATH}
