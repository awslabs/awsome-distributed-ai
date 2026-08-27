# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# DeepEP V2 image for AWS EFA clusters.
# See README.md for build and run instructions.

# CUDA 13.1 is the default. Earlier DeepEP revisions needed >= 13.1 on
# p6/Blackwell (ptxas 13.0.88 rejects the 32-bit st.bulk size operand); DeepEP
# main carries the 64-bit fix (amazon-contributing/DeepEP#3), so 13.0 works too.
ARG CUDA_VERSION=13.1.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG TORCH_VERSION=2.11.0
# Torch wheel index. Kept explicit rather than derived from nvcc: pytorch.org has
# no cu131 index, and cu130 wheels run on a 13.1 toolkit (same CUDA major).
ARG TORCH_CUDA_INDEX=cu130
ARG GDRCOPY_VERSION=v2.5.2

# EFA installer: provides the complete EFA userspace stack (OpenMPI, the EFA
# runtime, and the aws-ofi-nccl plugin). EFA-GDA requires installer >= 1.50;
# the build fails loudly on a too-old installer (plugin gate below).
# EFA_INSTALLER_TARBALL overrides the download with a local tarball staged next
# to this Dockerfile.
ARG EFA_INSTALLER_VERSION=1.50.0
ARG EFA_INSTALLER_TARBALL=

# NCCL: defaults to the public NVIDIA release. For internal use, override with
# the upstream fork + staging branch (see build instructions in README.md).
ARG NCCL_REPO=https://github.com/NVIDIA/nccl.git
ARG NCCL_REF=v2.31.2-1

# DeepEP V2. The repo is pinned to amazon-contributing/DeepEP inside
# setup_deepep_gin.sh; only the ref is overridable.
ARG DEEPEP_REF=main

# CUDA architecture(s), semicolon-separated:
# 9.0 = Hopper (H100/H200, sm_90); 10.0/10.3 = Blackwell (B200/B300, sm_100/sm_103).
# Defaults to Hopper + Blackwell so one image runs on p5/p5en and p6.
ARG TORCH_CUDA_ARCH_LIST="9.0;10.0"
ARG NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100"
ARG CUDA_HOME="/usr/local/cuda"

ENV DEBIAN_FRONTEND=noninteractive

## Remove conflicting distro packages
RUN apt-get update -y \
    && apt-get remove -y --allow-change-held-packages \
    ibverbs-utils \
    libibverbs-dev \
    libibverbs1 \
    libmlx5-1 \
    libnccl2 \
    libnccl-dev || true
RUN rm -rf /opt/hpcx /usr/local/mpi /etc/ld.so.conf.d/hpcx.conf && ldconfig

## Toolchain
RUN apt-get update -y && apt-get install -y \
    apt-utils autoconf automake build-essential check cmake curl debhelper \
    devscripts git gcc gdb libsubunit-dev libtool ninja-build meson pandoc \
    pkg-config vim wget \
    python3.10-dev python3.10-venv python3-distutils \
    cython3 \
    libnl-3-dev libnl-route-3-dev libudev-dev libsystemd-dev \
    libhwloc-dev \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get purge -y cuda-compat-* || true

RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3 /tmp/get-pip.py \
    && pip3 install awscli nvidia-ml-py ninja Cython

## GDRCopy
# Required by aws-ofi-nccl: the plugin's GIN initialization opens a gdr handle
# regardless of backend, so building or running without GDRCopy leaves the GIN
# backends unable to initialize (verified: DeepEP then fails with "NCCL GIN is
# unavailable"). Built from source because no distro package ships it. The
# matching gdrdrv kernel module must be loaded on the host (see README).
ARG GDRCOPY_PREFIX="/opt/gdrcopy"
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix="${GDRCOPY_PREFIX}" install \
    && rm -rf /tmp/gdrcopy
ENV LD_LIBRARY_PATH="${GDRCOPY_PREFIX}/lib:$LD_LIBRARY_PATH"
ENV LIBRARY_PATH="${GDRCOPY_PREFIX}/lib:$LIBRARY_PATH"
ENV PATH="${GDRCOPY_PREFIX}/bin:$PATH"

## EFA installer (OpenMPI + EFA runtime; local tarball wins when set)
RUN --mount=type=bind,target=/ctx \
    apt-get update -y \
    && cd /tmp \
    && if [ -n "${EFA_INSTALLER_TARBALL}" ]; then \
         echo "=== EFA installer: LOCAL tarball /ctx/${EFA_INSTALLER_TARBALL} ==="; \
         if [ ! -f "/ctx/${EFA_INSTALLER_TARBALL}" ]; then \
           echo "ERROR: ${EFA_INSTALLER_TARBALL} not found in build context" >&2; exit 1; \
         fi; \
         tar -xf "/ctx/${EFA_INSTALLER_TARBALL}"; \
       else \
         echo "=== EFA installer: downloading v${EFA_INSTALLER_VERSION} ==="; \
         curl -fsSL -O "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz" \
         && tar -xf "aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"; \
       fi \
    && cd aws-efa-installer \
    && ./efa_installer.sh --disable-ngc -y --skip-kmod --skip-limit-conf --no-verify \
    && ldconfig \
    && { nm -D /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so 2>/dev/null | grep -qw ncclGinPlugin_v14 \
         && echo "OK: bundled plugin is EFA-GDA-capable (ncclGinPlugin_v14 present)" \
         || { echo "ERROR: installer's plugin has no ncclGinPlugin_v14 export: EFA-GDA requires installer >= 1.50" >&2; exit 1; }; } \
    && rm -rf /tmp/aws-efa-installer* /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:$PATH
ENV EFA_PREFIX=/opt/amazon/efa

## NCCL
ENV NCCL_HOME=/opt/nccl/build
RUN git clone ${NCCL_REPO} /opt/nccl \
    && cd /opt/nccl \
    && git checkout ${NCCL_REF} \
    && make -j"$(nproc)" src.build CUDA_HOME="${CUDA_HOME}" NVCC_GENCODE="${NVCC_GENCODE}" \
    && test -f ${NCCL_HOME}/include/nccl_device.h \
       || (echo "ERROR: nccl_device.h missing -- this NCCL is not GIN-capable" >&2 && exit 1)
ENV LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH}"

## aws-ofi-nccl: the EFA installer (>= 1.50) bundles the plugin with EFA-GDA
## support; the installer layer above gates on the plugin's ncclGinPlugin_v14
## export (v11/v13 op-tables are exported by every build; the v14 table is
## EFA-GDA-only), so a too-old installer fails that layer immediately.
ENV OFI_HOME=/opt/amazon/ofi-nccl
ENV LD_LIBRARY_PATH="${OFI_HOME}/lib:${LD_LIBRARY_PATH}"
ENV NCCL_NET_PLUGIN="${OFI_HOME}/lib/libnccl-net-ofi.so"
ENV NCCL_GIN_PLUGIN="${OFI_HOME}/lib/libnccl-net-ofi.so"

# Keep NCCL's bootstrap off the docker/loopback/veth interfaces.
ENV NCCL_SOCKET_IFNAME=^docker,lo,veth

## PyTorch.
# The torch wheel drags in pip's stock NCCL (nvidia-nccl-cu*), which predates GIN.
# We remove it rather than keep both: DeepEP refuses to import when it detects two
# NCCL runtimes, and with both on the loader path which one wins is load-order
# luck. The GIN capability itself is enforced above at NCCL build time
# (nccl_device.h check errors the build), so this cannot silently downgrade: the
# image ends up with exactly one NCCL, the GIN-capable one it was built and
# tested with.
RUN pip3 install torch==${TORCH_VERSION} numpy --index-url https://download.pytorch.org/whl/${TORCH_CUDA_INDEX} \
    && pip3 uninstall -y nvidia-nccl-cu13 nvidia-nccl-cu12 nvidia-nccl 2>/dev/null || true

## No NVSHMEM: DeepEP V2's transport is NCCL GIN, so the NVSHMEM backend is not
## built (the DeepEP fork's build only links NVSHMEM when NVSHMEM_DIR points at an
## install). To additionally build the legacy NVSHMEM backend, install the
## libnvshmem3-*-cuda-13 packages and set NVSHMEM_DIR before running the setup script.

## DeepEP V2.
ARG DEEPEP_PREFIX="/opt/amazon/deepep"
RUN --mount=type=bind,source=setup_deepep_gin.sh,target=/tmp/setup_deepep_gin.sh \
    set -e; \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    /tmp/setup_deepep_gin.sh \
        --deepep-ref "${DEEPEP_REF}" \
        --deepep-prefix "${DEEPEP_PREFIX}" \
        --nccl-root "${NCCL_HOME}"

## Runtime env
ENV NVIDIA_GDRCOPY=enabled
ENV NCCL_OFI_RDMA_GDR_FLUSH_DISABLE=0
WORKDIR /root
