# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

####################################################################################################
# This is a sample Dockerfile, with optional stanzas. Please read through this Dockerfile,
# understand what it does, then create your own Dockerfile.
#
# Sample build instructions:
#
#     docker build --progress=plain -t nvidia-pt-od:latest -f 0.nvcr-pytorch-aws.dockerfile .
#     rm /fsx/nvidia-pt-od__latest.sqsh ; enroot import -o /fsx/nvidia-pt-od__latest.sqsh dockerd://nvidia-pt-od:latest
#
# Compute nodes (aka build nodes) are transient, so we need to keep the docker image on shared fs,
# which head node can load into its local registry.
#
#     # Build node: save image to file
#     docker save nvidia-pt-od:latest > /fsx/nvidia-pt-od__latest.tar
#
#     # Load image to local docker registry -> on head node, or new compute/build node.
#     docker load < /fsx/nvidia-pt-od__latest.tar
####################################################################################################
FROM nvcr.io/nvidia/pytorch:24.09-py3
ENV DEBIAN_FRONTEND=noninteractive

# The three must-be-built packages.
# Efa-installer>=1.29.1 required for nccl>=2.19.0 to avoid libfabric NCCL error.
ENV EFA_INSTALLER_VERSION=1.35.0
ENV AWS_OFI_NCCL_VERSION=1.12.1-aws
ENV NCCL_TESTS_VERSION=master

## Uncomment below when this Dockerfile builds a container image with efa-installer<1.29.1 and
# nccl>=2.19.0. See https://github.com/awslabs/awsome-distributed-ai/tree/main/architectures/efa-cheatsheet.md
#ENV FI_EFA_SET_CUDA_SYNC_MEMOPS=0

RUN apt-get update -y
RUN apt-get remove -y --allow-change-held-packages \
                      libmlx5-1 ibverbs-utils libibverbs-dev libibverbs1

# We noticed that since 23.09, we can't just delete the whole /opt/hpcx/, otherwise `import torch`
# complains about missing libuc?.so.
RUN rm -rf /opt/hpcx/ompi \
    && rm -rf /usr/local/mpi \
    && rm -rf /opt/hpcx/nccl_rdma_sharp_plugin \
    && ldconfig
ENV OPAL_PREFIX=
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    git \
    gcc \
    vim \
    kmod \
    openssh-client \
    openssh-server \
    build-essential \
    curl \
    autoconf \
    libtool \
    gdb \
    automake \
    cmake \
    apt-utils \
    libhwloc-dev \
    aptitude && \
    DEBIAN_FRONTEND=noninteractive apt autoremove -y

# EFA
RUN apt-get update && \
    cd /tmp && \
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz  && \
    tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    # ONLY add `--skip-kmod`, `--no-verify` and `--skip-limit-conf` flags to container image.
    # Those three flags must NOT be used on the host.
    #
    # Explanations:
    # - to build EFA in the Dockerfile, we added --skip-kmod and --no-verify. Without these flags,
    #   the Dockerfile will fail to build. If installing EFA on the host and not in a container,
    #   please remove these flags.
    # - The --skip-limit-conf can be retained in Dockerfile, but it's redundant as the host already
    #   has these limits set by efa_installer.
    ./efa_installer.sh -y -g -d --skip-kmod --no-verify --skip-limit-conf && \
    ldconfig && \
    rm -rf /tmp/aws-efa-installer /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH


####################################################################################################
# [CUSTOM_NCCL_OPTION_1] Uncomment below stanza to install another NCCL version using the official
# binaries.
#
# NCCL EFA plugin (aws-ofi-nccl) depends on mpi, hence we must rebuild openmpi before building the
# aws-ofi-ccnl.
####################################################################################################
#ENV NCCL_VERSION=2.19.3-1
#RUN cd /opt && \
#    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
#    dpkg -i cuda-keyring_1.0-1_all.deb && \
#    apt update && \
#    apt install -y libnccl2==${NCCL_VERSION} libnccl-dev==${NCCL_VERSION} && \
#    echo NCCL_SOCKET_IFNAME=^docker0,lo >> /etc/nccl.conf


####################################################################################################
# [CUSTOM_NCCL_OPTION_2] Install NCCL from source to the same location as the built-in ones. The
# benefits of installing to the same location as the built-in version are:
#
# 1. There's only ever a single libnccl version offered by this image, preventing application from
#    mistakenly chooses a wrong version.
# 2. No longer needing extra settings for LD_LIBRARY_PATH or LD_PRELOAD.
#
# NCCL EFA plugin (aws-ofi-nccl) depends on mpi, hence we must rebuild openmpi before building the
# aws-ofi-ccnl.
####################################################################################################
ENV NCCL_VERSION=2.19.3-1
RUN apt-get remove -y libnccl2 libnccl-dev \
   && cd /tmp \
   && git clone https://github.com/NVIDIA/nccl.git -b v${NCCL_VERSION} \
   && cd nccl \
   && make -j src.build BUILDDIR=/usr \
   # Build for p4 & p5.
   NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90, -gencode=arch=compute_80,code=sm_80" \
   && rm -rf /tmp/nccl \
   && echo NCCL_SOCKET_IFNAME=^docker0,lo >> /etc/nccl.conf


####################################################################################################
# Rebuild OpenMPI with custom PMIX version. E.g., to match what host's Slurm is built with (see
# /opt/pmix/ on host, or run pmix_info on host).
#
# May be needed on rare occassions when `srun --mpi=pmix --container-image=... <mpi_application>`
# mysteriously crashes.
#
# NCCL EFA plugin (aws-ofi-nccl) depends on mpi, hence we must rebuild openmpi before building the
# aws-ofi-ccnl.
####################################################################################################
ENV OPEN_MPI_PATH=/opt/amazon/openmpi

# OpenMPI build script claims PMIX_VERSION, and complains if we use it.
ENV CUSTOM_PMIX_VERSION=4.2.6
RUN apt-get update && apt-get install -y libevent-dev \
    && cd /tmp \
    && wget https://github.com/openpmix/openpmix/releases/download/v${CUSTOM_PMIX_VERSION}/pmix-${CUSTOM_PMIX_VERSION}.tar.gz \
    && tar -xzf pmix-${CUSTOM_PMIX_VERSION}.tar.gz \
    && rm pmix-${CUSTOM_PMIX_VERSION}.tar.gz \
    && cd pmix-${CUSTOM_PMIX_VERSION}/ \
    && ./autogen.pl \
    && ./configure --prefix=/opt/pmix \
    && make -j \
    && make install \
    && echo /opt/pmix/lib > /etc/ld.so.conf.d/pmix.conf \
    && ldconfig \
    && cd / \
    && rm -fr /tmp/pmix-${CUSTOM_PMIX_VERSION}/
# To silence this runtime error message:
# [p4de-st-p4de-2:110912] PMIX ERROR: ERROR in file gds_ds12_lock_pthread.c at line 168
ENV PMIX_GDS_MODULE=^ds12 \
    PMIX_MCA_gds=^ds12

# Rebuild openmpi with DLC style (which it remarks as "without libfabric"), with the above pmix.
ENV OMPI_VERSION=4.1.6
RUN rm -fr ${OPEN_MPI_PATH} \
 && mkdir /tmp/openmpi \
 && cd /tmp/openmpi \
 && wget --quiet https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OMPI_VERSION}.tar.gz \
 && tar zxf openmpi-${OMPI_VERSION}.tar.gz \
 && rm openmpi-${OMPI_VERSION}.tar.gz \
 && cd openmpi-${OMPI_VERSION} \
 && ./configure --enable-orterun-prefix-by-default --prefix=$OPEN_MPI_PATH --with-cuda=${CUDA_HOME} --with-slurm --with-pmix=/opt/pmix \
 && make -j $(nproc) all \
 && make install \
 && ldconfig \
 && cd / \
 && rm -rf /tmp/openmpi \
 && ompi_info --parsable --all | grep mpi_built_with_cuda_support:value \
 # Verify pmix from /opt/pmix/
 && ldd /opt/amazon/openmpi/lib/openmpi/mca_pmix_ext3x.so | grep '/opt/pmix/lib/libpmix.so.* ' > /opt/amazon/openmpi-pmix.txt
####################################################################################################


# NCCL EFA Plugin
RUN mkdir -p /tmp && \
    cd /tmp && \
    curl -LO https://github.com/aws/aws-ofi-nccl/archive/refs/tags/v${AWS_OFI_NCCL_VERSION}.tar.gz && \
    tar -xzf /tmp/v${AWS_OFI_NCCL_VERSION}.tar.gz && \
    rm /tmp/v${AWS_OFI_NCCL_VERSION}.tar.gz && \
    mv aws-ofi-nccl-${AWS_OFI_NCCL_VERSION} aws-ofi-nccl && \
    cd /tmp/aws-ofi-nccl && \
    ./autogen.sh && \
    ./configure --prefix=/opt/amazon/efa \
        --with-libfabric=/opt/amazon/efa \
        --with-cuda=/usr/local/cuda \
        --enable-platform-aws \
        --with-mpi=/opt/amazon/openmpi && \
    make -j$(nproc) install && \
    rm -rf /tmp/aws-ofi/nccl

# Do this to minimize the ld path env vars that users need to define when running this image.
RUN echo "/usr/local/lib"      >> /etc/ld.so.conf.d/local.conf && \
    echo "/opt/amazon/openmpi/lib" >> /etc/ld.so.conf.d/efa.conf && \
    ldconfig

ENV OMPI_MCA_pml=^cm,ucx            \
    OMPI_MCA_btl=tcp,self           \
    OMPI_MCA_btl_tcp_if_exclude=lo,docker0 \
    OPAL_PREFIX=/opt/amazon/openmpi \
    # https://discuss.pytorch.org/t/nccl-network-is-unreachable-connection-refused-when-initializing-ddp/137352
    # https://github.com/pytorch/pytorch/issues/68893
    NCCL_SOCKET_IFNAME=^docker,lo

ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# NCCL-tests: always good to include this as a diagnostic tool.
RUN git clone https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && cd /opt/nccl-tests \
    && git checkout ${NCCL_TESTS_VERSION} \
    && make MPI=1 \
    MPI_HOME=/opt/amazon/openmpi \
    CUDA_HOME=/usr/local/cuda \
    NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_80,code=sm_80"


####################################################################################################
# Custom packages. Disable as you like. NOTE: always check `pip list` what's been installed. For
# example, the base container comes pre-installed with Transformer Engine, flash attention, triton
# (https://github.com/openai/triton/), etc.
####################################################################################################
# Install the xformers dependency from source, because pip install either breaks or try to pull
# its own pt + cuda.
#
# Pre-requisite: build node has enough memory to compile xformers. More info on the stanza.
RUN export TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX" && \
    # On p4de.24xlarge:
    # - MAX_JOBS=16 => 145GB memory
    # - MAX_JOBS=32 => 241GB memory
    # - MAX_JOBS=48 => 243GB memory, 542.5s
    #
    # NOTE: must export MAX_JOBS. For some reason, `MAX_JOBS=16 pip install ...` doesn't seem to
    #       work to prevent OOM.
    export MAX_JOBS=32 && \
    export NVCC_PREPEND_FLAGS="-t 32" && \
    pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
