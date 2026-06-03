# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

ARG AWS_REGION=us-west-2

FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-ec2

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libpmix-dev libpmix2

# reinstall EFA, to restore the openmpi
ENV EFA_INSTALLER_VERSION=latest
RUN apt-get update && \
    cd /tmp && \
    rm -fr /opt/amazon/openmpi && \
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz  && \
    tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf && \
    ldconfig && \
    rm -rf /tmp/aws-efa-installer /var/lib/apt/lists/* && \
    /opt/amazon/efa/bin/fi_info --version

# Repeat this from base Dockefile
# Install AWS OFI NCCL plug-in
ENV AWS_OFI_NCCL_VERSION=1.7.1
RUN apt-get update && apt-get install -y autoconf
RUN mkdir /tmp/efa-ofi-nccl \
 && cd /tmp/efa-ofi-nccl \
 && git clone https://github.com/aws/aws-ofi-nccl.git -b v${AWS_OFI_NCCL_VERSION}-aws \
 && cd aws-ofi-nccl \
 && ./autogen.sh \
 && ./configure --with-libfabric=/opt/amazon/efa \
  --with-mpi=/opt/amazon/openmpi \
  --with-cuda=/opt/conda --prefix=/usr/local \
 && make \
 && make install \
 && rm -rf /tmp/efa-ofi-nccl \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/efa/bin:$PATH

RUN cd /opt && git clone https://github.com/NVIDIA/nccl-tests \
    && cd nccl-tests \
    && make MPI=1 MPI_HOME=/opt/amazon/openmpi


#################################################
## Install NVIDIA GDRCopy
RUN apt-get update && apt-get install -y check libsubunit0 libsubunit-dev pkg-config \
    && git clone https://github.com/NVIDIA/gdrcopy.git /opt/gdrcopy \
    && cd /opt/gdrcopy \
    && CUDA=/opt/conda make lib_install \
    # Optional: tests tool. Need to point to the stub libcuda.so
    # See: https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.7.1/ubuntu2004/devel/Dockerfile#L68
    && LIBRARY_PATH=/opt/conda/lib/stubs CUDA=/opt/conda make exes_install
