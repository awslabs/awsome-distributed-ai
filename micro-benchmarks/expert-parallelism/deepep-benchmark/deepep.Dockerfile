# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
ARG CUDA_VERSION=13.0.2

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG TARGETARCH

ARG GDRCOPY_VERSION=v2.5.2
ARG EFA_INSTALLER_VERSION=1.48.0
ARG NCCL_VERSION=v2.30.4-1
ARG NCCL_TESTS_VERSION=v2.18.3
ARG NVSHMEM_VERSION=3.7.0
ARG TORCH_VERSION=2.11.0
ARG DEEPEP_COMMIT=567632d

# CUDA architecture(s), semicolon-separated:
# 9.0 = Hopper (H100, sm_90), 10.0 = Blackwell (B200/B300, sm_100). Defaults to
# both so one image runs on Hopper and Blackwell; override with e.g.
# --build-arg GPU_ARCH=90 \
# --build-arg TORCH_CUDA_ARCH_LIST=9.0 \
# --build-arg NVCC_GENCODE=-gencode=arch=compute_90,code=sm_90 to build a smaller Hopper-only image.
ARG GPU_ARCH="90;100"
ARG TORCH_CUDA_ARCH_LIST="9.0;10.0"
ARG NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100"

ARG CUDA_HOME="/usr/local/cuda"

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get remove -y --allow-change-held-packages \
    ibverbs-utils \
    libibverbs-dev \
    libibverbs1 \
    libmlx5-1 \
    libnccl2 \
    libnccl-dev

RUN rm -rf /opt/hpcx \
    && rm -rf /usr/local/mpi \
    && rm -f /etc/ld.so.conf.d/hpcx.conf \
    && ldconfig

ENV OPAL_PREFIX=

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils \
    autoconf \
    automake \
    build-essential \
    check \
    cmake \
    curl \
    debhelper \
    devscripts \
    git \
    gcc \
    gdb \
    libsubunit-dev \
    libtool \
    openssh-client \
    openssh-server \
    pkg-config \
    python3-distutils \
    vim \
    python3.10-dev \
    python3.10-venv
RUN apt-get purge -y cuda-compat-*

RUN mkdir -p /var/run/sshd
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3 /tmp/get-pip.py \
    && pip3 install awscli nvidia-ml-py setuptools wheel build ninja

#################################################
## Install NVIDIA GDRCopy
##
## NOTE: if `nccl-tests` or `/opt/gdrcopy/bin/sanity -v` crashes with incompatible version, ensure
## that the cuda-compat-xx-x package is the latest.
ARG GDRCOPY_PREFIX="/opt/gdrcopy"
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix="${GDRCOPY_PREFIX}" install \
    && rm -rf /tmp/gdrcopy

ENV LD_LIBRARY_PATH="${GDRCOPY_PREFIX}/lib:$LD_LIBRARY_PATH"
ENV LIBRARY_PATH="${GDRCOPY_PREFIX}/lib:$LIBRARY_PATH"
ENV PATH="${GDRCOPY_PREFIX}/bin:$PATH"

#################################################
## Install EFA installer
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && if printf '%s\n' "1.47.0" "${EFA_INSTALLER_VERSION}" | sort -V | head -n1 | grep -qx "${EFA_INSTALLER_VERSION}"; then \
        ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify; \
    else \
        ./efa_installer.sh --disable-build-ngc --disable-ngc -y --skip-kmod --skip-limit-conf --no-verify; \
    fi \
    && ldconfig \
    && rm -rf $HOME/aws-efa-installer

ENV LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:$PATH

###################################################
## Install NCCL
RUN git clone -b ${NCCL_VERSION} https://github.com/NVIDIA/nccl.git  /opt/nccl \
    && cd /opt/nccl \
    && make -j $(nproc) src.build CUDA_HOME="${CUDA_HOME}"

ENV LD_LIBRARY_PATH=/opt/nccl/build/lib:$LD_LIBRARY_PATH

###################################################
## Install NCCL-tests
RUN git clone -b ${NCCL_TESTS_VERSION} https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && cd /opt/nccl-tests \
    && make -j $(nproc) \
    MPI=1 \
    MPI_HOME=/opt/amazon/openmpi/ \
    CUDA_HOME="${CUDA_HOME}" \
    NCCL_HOME=/opt/nccl/build

###################################################
## Install NVSHMEM
ARG NVSHMEM_PREFIX="/opt/nvshmem"
ARG LIBFABRIC_HOME="/opt/amazon/efa"

RUN CUDA_VERSION_MAJOR=$(nvcc --version | grep -oP 'release \K[0-9]+') && \
    if [ "${TARGETARCH}" = "arm64" ]; then NVSHMEM_ARCH="linux-sbsa"; else NVSHMEM_ARCH="linux-x86_64"; fi && \
    NVSHMEM_ARCHIVE="libnvshmem-${NVSHMEM_ARCH}-${NVSHMEM_VERSION}_cuda${CUDA_VERSION_MAJOR}-archive.tar.xz" && \
    NVSHMEM_DOWNLOAD_URL="https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/${NVSHMEM_ARCH}/${NVSHMEM_ARCHIVE}" && \
    curl -fSL "${NVSHMEM_DOWNLOAD_URL}" -o "/tmp/${NVSHMEM_ARCHIVE}" && \
    mkdir -p "$NVSHMEM_PREFIX" && \
    tar -xf "/tmp/${NVSHMEM_ARCHIVE}" --strip-components=1 -C "$NVSHMEM_PREFIX" && \
    rm "/tmp/${NVSHMEM_ARCHIVE}"

# Uncomment to build NVSHMEM from source
# RUN git clone https://github.com/NVIDIA/nvshmem.git /tmp/nvshmem && \
#     cd /tmp/nvshmem && \
#     git checkout "v${NVSHMEM_VERSION}-0" && \
#     cd /tmp/nvshmem && \
#     rm -rf build && mkdir build && cd build && \
#     cmake .. \
#         -DCMAKE_INSTALL_PREFIX="${NVSHMEM_PREFIX}" \
#         -DCUDA_HOME="${CUDA_HOME}" \
#         -DLIBFABRIC_HOME="${LIBFABRIC_HOME}" \
#         -DNVSHMEM_LIBFABRIC_SUPPORT=ON \
#         -DNVSHMEM_MPI_SUPPORT=OFF \
#         -DNVSHMEM_IBRC_SUPPORT=OFF \
#         -DNVSHMEM_IBGDA_SUPPORT=OFF \
#         -DNVSHMEM_IBDEVX_SUPPORT=OFF \
#         -DNVSHMEM_UCX_SUPPORT=OFF \
#         -DNVSHMEM_SHMEM_SUPPORT=OFF \
#         -DNVSHMEM_PMIX_SUPPORT=OFF \
#         -DNVSHMEM_USE_NCCL=OFF \
#         -DNVSHMEM_USE_GDRCOPY=ON \
#         -DGDRCOPY_HOME="${GDRCOPY_PREFIX}" \
#         -DNVSHMEM_USE_MLX5DV=OFF \
#         -DNVSHMEM_BUILD_TESTS=OFF \
#         -DNVSHMEM_BUILD_EXAMPLES=OFF \
#         -DNVSHMEM_BUILD_PYTHON_LIB=OFF \
#         -DNVSHMEM_BUILD_BITCODE_LIBRARY=OFF \
#         -DCMAKE_CUDA_ARCHITECTURES="${GPU_ARCH}" && \
#     make -j"$(nproc)" && \
#     make install && \
#     rm -rf /tmp/nvshmem

ENV LD_LIBRARY_PATH="${NVSHMEM_PREFIX}/lib:${LD_LIBRARY_PATH}"
ENV PATH="${NVSHMEM_PREFIX}/bin:${PATH}"

ENV NVSHMEM_REMOTE_TRANSPORT=libfabric
ENV NVSHMEM_LIBFABRIC_PROVIDER=efa
ENV NVSHMEM_NETDEVS_POLICY=EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE

RUN rm -rf /var/lib/apt/lists/*

## Set Open MPI variables to exclude network interface and conduit.
ENV OMPI_MCA_pml=^ucx            \
    OMPI_MCA_btl=tcp,self           \
    OMPI_MCA_btl_tcp_if_exclude=lo,docker0,veth_def_agent\
    OPAL_PREFIX=/opt/amazon/openmpi \
    NCCL_SOCKET_IFNAME=^docker,lo,veth

## Turn off PMIx Error https://github.com/open-mpi/ompi/issues/7516
ENV PMIX_MCA_gds=hash

## Set LD_PRELOAD for NCCL library
ENV LD_PRELOAD=/opt/nccl/build/lib/libnccl.so

################################ PyTorch ########################################
RUN CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | tr -d '.') && \
    CUDA_MAJOR=$(nvcc --version | grep -oP 'release \K[0-9]+') && \
    pip3 install torch==${TORCH_VERSION} numpy --index-url https://download.pytorch.org/whl/cu${CUDA_VER} && \
    pip3 uninstall -y nvidia-nvshmem-cu${CUDA_MAJOR}

################################ DeepEP ########################################
ARG DEEPEP_PREFIX="/DeepEP"

RUN --mount=type=bind,source=deepep_aws_efa.patch,target=/tmp/deepep_aws/deepep_aws_efa.patch \
    if [ "${TORCH_CUDA_ARCH_LIST}" = "9.0" ]; then DISABLE_AGGRESSIVE_PTX_INSTRS=0; else DISABLE_AGGRESSIVE_PTX_INSTRS=1; fi && \
    export DISABLE_AGGRESSIVE_PTX_INSTRS && \
    git clone https://github.com/deepseek-ai/DeepEP.git ${DEEPEP_PREFIX} && \
    cd ${DEEPEP_PREFIX} && \
    git checkout ${DEEPEP_COMMIT} && \
    patch -p1 < /tmp/deepep_aws/deepep_aws_efa.patch && \
    CUDA_VERSION_MAJOR=$(nvcc --version | grep -oP 'release \K[0-9]+') && \
    if [ "${CUDA_VERSION_MAJOR}" -ge 13 ]; then \
        sed -i "s|f'{nvshmem_dir}/include']|f'{nvshmem_dir}/include', '${CUDA_HOME}/include/cccl']|" "setup.py"; \
    fi && \
    NVSHMEM_DIR="${NVSHMEM_PREFIX}" pip3 install --no-build-isolation .
