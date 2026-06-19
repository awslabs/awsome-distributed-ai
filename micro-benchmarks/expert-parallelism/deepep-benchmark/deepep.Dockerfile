# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

################################ NCCL ########################################

ARG GDRCOPY_VERSION=v2.5.2
ARG EFA_INSTALLER_VERSION=1.48.0
ARG NCCL_VERSION=v2.30.4-1
ARG NCCL_TESTS_VERSION=v2.18.3

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

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
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
    kmod \
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

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH

RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3 /tmp/get-pip.py \
    && pip3 install awscli pynvml

#################################################
## Install NVIDIA GDRCopy
##
## NOTE: if `nccl-tests` or `/opt/gdrcopy/bin/sanity -v` crashes with incompatible version, ensure
## that the cuda-compat-xx-x package is the latest.
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix=/opt/gdrcopy install

ENV LD_LIBRARY_PATH=/opt/gdrcopy/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/gdrcopy/lib:$LIBRARY_PATH
ENV PATH=/opt/gdrcopy/bin:$PATH

#################################################
## Install EFA installer
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf $HOME/aws-efa-installer

###################################################
## Install NCCL
RUN git clone -b ${NCCL_VERSION} https://github.com/NVIDIA/nccl.git  /opt/nccl \
    && cd /opt/nccl \
    && make -j $(nproc) src.build CUDA_HOME=/usr/local/cuda \
    NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100"

###################################################
## Install NCCL-tests
RUN git clone -b ${NCCL_TESTS_VERSION} https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && cd /opt/nccl-tests \
    && make -j $(nproc) \
    MPI=1 \
    MPI_HOME=/opt/amazon/openmpi/ \
    CUDA_HOME=/usr/local/cuda \
    NCCL_HOME=/opt/nccl/build \
    NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100"

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

################################ venv ################################

RUN python3 -m venv /opt/deepep \
    && /opt/deepep/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel

ENV PATH=/opt/deepep/bin:$PATH
ENV VIRTUAL_ENV=/opt/deepep

################################ NVSHMEM ########################################

ARG NVSHMEM_VERSION=v3.7.0-0

ENV NVSHMEM_SRC=/opt/nvshmem_src
RUN git clone -b ${NVSHMEM_VERSION} https://github.com/NVIDIA/nvshmem.git ${NVSHMEM_SRC}

COPY ./setup_deepep_efa.sh /tmp/setup_deepep_efa.sh

ENV NVSHMEM_DIR=${NVSHMEM_SRC}/install
ENV NVSHMEM_HOME=${NVSHMEM_DIR}
ENV PATH=${NVSHMEM_DIR}/bin:$PATH
ENV LD_LIBRARY_PATH=${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH

ENV NVSHMEM_REMOTE_TRANSPORT=libfabric 
ENV NVSHMEM_LIBFABRIC_PROVIDER=efa

################################ PyTorch ########################################

RUN /opt/deepep/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128 \
    && /opt/deepep/bin/pip uninstall -y nvidia-nvshmem-cu12 \
    && /opt/deepep/bin/pip install ninja numpy cmake pytest

################################ DeepEP ########################################

ARG DEEPEP_COMMIT=567632d

# CUDA architecture(s) to build NVSHMEM and DeepEP for, semicolon-separated:
# 90 = Hopper (H100, sm_90), 100 = Blackwell (B200/B300, sm_100). Defaults to
# both so one image runs on Hopper and Blackwell; override with e.g.
# --build-arg GPU_ARCH=90 to build a smaller Hopper-only image.
ARG GPU_ARCH="90;100"

RUN git clone https://github.com/deepseek-ai/DeepEP.git /DeepEP \
    && cd /DeepEP \
    && git checkout ${DEEPEP_COMMIT}

RUN /tmp/setup_deepep_efa.sh \
    --cuda-home /usr/local/cuda \
    --libfabric-home /opt/amazon/efa \
    --gdrcopy-home /opt/gdrcopy \
    --gpu-arch "${GPU_ARCH}" \
    --venv /opt/deepep \
    --deepep-src /DeepEP \
    --nvshmem-src ${NVSHMEM_SRC}
