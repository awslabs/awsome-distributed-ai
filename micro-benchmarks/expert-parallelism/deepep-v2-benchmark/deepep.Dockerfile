# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
ARG CUDA_VERSION=13.0.2

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG GDRCOPY_VERSION=v2.5.2
ARG EFA_INSTALLER_VERSION=1.50.0
ARG DEEPEP_REF="main"
ARG NCCL_VERSION=2.31.2
ARG PYTHON_VERSION=3.10
ARG TORCH_VERSION=2.11.0
ARG TORCH_CUDA_INDEX=cu130

# CUDA architecture(s), semicolon-separated:
# 9.0 = Hopper (H100, sm_90), 10.0 and 10.3 = Blackwell (B200/B300, sm_100,sm_103). Defaults to
# both so one image runs on Hopper and Blackwell; override with e.g.
# --build-arg TORCH_CUDA_ARCH_LIST=9.0 \
# --build-arg NVCC_GENCODE=-gencode=arch=compute_90,code=sm_90 to build a smaller Hopper-only image.
ARG TORCH_CUDA_ARCH_LIST="9.0;10.0;10.3"
ARG NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_103,code=sm_103"

ARG CUDA_HOME="/usr/local/cuda"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get remove -y --allow-change-held-packages \
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

RUN apt-get install -y --no-install-recommends \
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
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv

RUN apt-get purge -y cuda-compat-*

RUN mkdir -p /var/run/sshd
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3 /tmp/get-pip.py \
    && pip3 install awscli nvidia-ml-py ninja

#################################################
## Install NVIDIA GDRCopy
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
RUN cd $HOME && \
    curl --retry 3 --retry-delay 2 -fsSL -o aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
        https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    apt-get update && \
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify && \
    cd .. && rm -rf aws-efa-installer* && \
    ldconfig

ENV LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:$PATH

RUN rm -rf /var/lib/apt/lists/*

## Set Open MPI variables to exclude network interface and conduit.
ENV OMPI_MCA_pml=^ucx            \
    OMPI_MCA_btl=tcp,self           \
    OMPI_MCA_btl_tcp_if_exclude=lo,docker0,veth_def_agent\
    OPAL_PREFIX=/opt/amazon/openmpi \
    NCCL_SOCKET_IFNAME=^docker,lo,veth

## Turn off PMIx Error https://github.com/open-mpi/ompi/issues/7516
ENV PMIX_MCA_gds=hash

################################ PyTorch ########################################
RUN pip3 install torch==${TORCH_VERSION} numpy --index-url https://download.pytorch.org/whl/${TORCH_CUDA_INDEX}

################################ DeepEP v2 ########################################
RUN --mount=type=bind,source=setup_deepep_gin.sh,target=/tmp/setup_deepep_gin.sh \
    pip3 uninstall -y nvidia-nccl-cu13 nvidia-nccl-cu12 nvidia-nccl 2>/dev/null || true && \
    CUDA_MAJOR=$(nvcc --version | grep -oP 'release \K[0-9]+') && \
    pip3 install --no-deps nvidia-nccl-cu${CUDA_MAJOR}==${NCCL_VERSION} && \
    /tmp/setup_deepep_gin.sh \
      --deepep-ref $DEEPEP_REF \
      --nccl-root "/usr/local/lib/python${PYTHON_VERSION}/dist-packages/nvidia/nccl"

ENV NVIDIA_GDRCOPY=enabled
ENV NCCL_NET_PLUGIN=ofi
ENV NCCL_TUNER_PLUGIN=ofi
