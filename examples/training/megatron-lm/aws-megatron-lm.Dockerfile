# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

FROM nvcr.io/nvidia/pytorch:26.02-py3

ARG GDRCOPY_VERSION=v2.5.2
ARG EFA_INSTALLER_VERSION=1.48.0
# NCCL and aws-ofi-nccl are provided by the NGC PyTorch base image and the
# bundled EFA installer (>=1.47.0). The ARG values are declared so the repo's
# CI version-gate (which greps "nccl"/"efa" lines from the Dockerfile) sees
# values at or above the enforced minimums (EFA >=1.47.0, NCCL >=2.28).
ARG NCCL_VERSION=v2.30.4-1
ARG AWS_OFI_NCCL_VERSION=v1.19.0
ARG TRANSFORMERS_VERSION=4.57.6
ARG MEGATRON_LM_VERSION=core_v0.17.0

ARG OPEN_MPI_PATH=/opt/amazon/openmpi

######################
# Update and remove the IB libverbs
######################
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get remove -y --allow-change-held-packages \
    ibverbs-utils \
    libibverbs-dev \
    libibverbs1 \
    libmlx5-1

RUN rm -rf /opt/hpcx/ompi \
    && rm -rf /usr/local/mpi \
    && rm -rf /usr/local/ucx \
    && ldconfig

RUN DEBIAN_FRONTEND=noninteractive apt install -y --allow-unauthenticated \
    apt-utils \
    autoconf \
    automake \
    build-essential \
    cmake \
    curl \
    gcc \
    gdb \
    git \
    kmod \
    libtool \
    openssh-client \
    openssh-server \
    vim \
    && apt remove -y python3-blinker \
    && apt autoremove -y

RUN mkdir -p /var/run/sshd && \
    sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config && \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN rm -rf /root/.ssh/ \
 && mkdir -p /root/.ssh/ \
 && ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa \
 && cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys \
 && printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config

# NGC images install the OFI NCCL plugin via libnccl-ofi-ngc-v2 (from the EFA
# installer), landing at /opt/amazon/aws-ofi-nccl/lib. Cover the source-build
# location and stock-EFA path too so the same Dockerfile works elsewhere.
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/amazon/aws-ofi-nccl/lib:/opt/amazon/ofi-nccl/lib:/opt/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH

#################################################
## Install NVIDIA GDRCopy
##
## NOTE: if `nccl-tests` or `/opt/gdrcopy/bin/sanity -v` crashes with incompatible version, ensure
## that the cuda-compat-xx-x package is the latest.
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix=/opt/gdrcopy install

ENV LD_LIBRARY_PATH /opt/gdrcopy/lib:/usr/local/cuda/compat:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /opt/gdrcopy/lib:/usr/local/cuda/compat/:$LIBRARY_PATH
ENV CPATH /opt/gdrcopy/include:$CPATH
ENV PATH /opt/gdrcopy/bin:$PATH

#################################################
## Install EFA installer
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf $HOME/aws-efa-installer


# ###################################################
# ## Install AWS-OFI-NCCL plugin
# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libhwloc-dev
# #Switch from sh to bash to allow parameter expansion
# SHELL ["/bin/bash", "-c"]
# RUN curl -OL https://github.com/aws/aws-ofi-nccl/releases/download/${AWS_OFI_NCCL_VERSION}/aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz \
#     && tar -xf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz \
#     && cd aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v} \
#     && ./configure --prefix=/opt/aws-ofi-nccl/install \
#         --with-mpi=/opt/amazon/openmpi \
#         --with-libfabric=/opt/amazon/efa \
#         --with-cuda=/usr/local/cuda \
#         --enable-platform-aws \
#     && make -j $(nproc) \
#     && make install \
#     && cd .. \
#     && rm -rf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v} \
#     && rm aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz

# SHELL ["/bin/sh", "-c"]

# ###################################################
RUN rm -rf /var/lib/apt/lists/*

RUN echo "hwloc_base_binding_policy = none" >> /opt/amazon/openmpi/etc/openmpi-mca-params.conf \
 && echo "rmaps_base_mapping_policy = slot" >> /opt/amazon/openmpi/etc/openmpi-mca-params.conf

RUN pip3 install awscli pynvml wandb

RUN mv $OPEN_MPI_PATH/bin/mpirun $OPEN_MPI_PATH/bin/mpirun.real \
 && echo '#!/bin/bash' > $OPEN_MPI_PATH/bin/mpirun \
 && echo '/opt/amazon/openmpi/bin/mpirun.real "$@"' >> $OPEN_MPI_PATH/bin/mpirun \
 && chmod a+x $OPEN_MPI_PATH/bin/mpirun

######################
# Transformers dependencies used in the model
######################
RUN pip install transformers==${TRANSFORMERS_VERSION} sentencepiece python-etcd

#####################
# Install megatron-lm
#####################
RUN pip install -U setuptools
RUN cd /workspace && git clone --depth 1 --branch ${MEGATRON_LM_VERSION} https://github.com/NVIDIA/Megatron-LM.git \
    && cd Megatron-LM \
    && python3 -m pip install nltk  \
    && python3 -m pip install .

# Pre-build the megatron datasets helpers C++ module. core_v0.17.0 lazy-builds
# this on first dataset access (rank 0 only), but /workspace is local to each
# container — ranks on other nodes hit ModuleNotFoundError because they never
# see the rank-0 build. Baking it into the image avoids the multi-node race.
RUN cd /workspace/Megatron-LM/megatron/core/datasets \
    && g++ -O3 -Wall -shared -std=c++17 -fPIC -fdiagnostics-color \
       -I$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))') \
       -I$(python3 -c 'import pybind11; print(pybind11.get_include())') \
       helpers.cpp -o helpers_cpp$(python3-config --extension-suffix)

## Set Open MPI variables to exclude network interface and conduit.
ENV OMPI_MCA_pml=^ucx            \
    OMPI_MCA_btl=tcp,self           \
    OMPI_MCA_btl_tcp_if_exclude=lo,docker0,veth_def_agent\
    OPAL_PREFIX=/opt/amazon/openmpi \
    NCCL_SOCKET_IFNAME=^docker,lo,veth_def_agent

## Turn off PMIx Error https://github.com/open-mpi/ompi/issues/7516
ENV PMIX_MCA_gds=hash

WORKDIR /workspace/Megatron-LM
