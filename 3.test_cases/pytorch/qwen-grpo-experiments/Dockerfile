# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Dockerfile for Qwen2.5-7B GRPO Experiments
# veRL v0.6.1 + EFA + vLLM + Ray on EKS (g6e.48xlarge / L40S)
#
# No model-specific patches required (Qwen is natively supported by vLLM/veRL)

FROM hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0

# EFA configuration
ARG OPEN_MPI_PATH=/opt/amazon/openmpi/
ENV EFA_VERSION=1.43.3

# Install system dependencies including EFA requirements
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    ninja-build \
    autoconf \
    build-essential \
    pciutils \
    environment-modules \
    tcl \
    tcl-dev \
    libnl-3-dev \
    libnl-route-3-dev \
    dmidecode \
    ethtool \
    iproute2 \
    libevent-dev \
    libhwloc-dev \
    openssh-server \
    openssh-client \
    systemd \
    udev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

#################################################
## Clean up HPC-X to avoid conflicts with EFA
RUN rm -rf /opt/hpcx \
    && rm -rf /usr/local/mpi \
    && rm -f /etc/ld.so.conf.d/hpcx.conf \
    && ldconfig

#################################################
## EFA SETUP
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_VERSION}.tar.gz \
    && tar -xf aws-efa-installer-${EFA_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify

# Environment paths for EFA
ENV PATH="/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# OpenMPI configuration
ENV OMPI_MCA_pml=^ucx
ENV OMPI_MCA_btl=tcp,self
ENV OMPI_MCA_btl_tcp_if_exclude=lo,docker0,veth_def_agent
ENV OPAL_PREFIX=/opt/amazon/openmpi

# EFA/NCCL configuration
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1
ENV FI_EFA_FORK_SAFE=1
ENV FI_EFA_ENABLE_SHM_TRANSFER=1
ENV NCCL_PROTO=simple
ENV NCCL_NET_GDR_LEVEL=LOC
ENV NCCL_SOCKET_IFNAME=^docker,lo,veth
ENV NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-ofi-tuner.so
ENV PMIX_MCA_gds=hash

#################################################
## NCCL tests for verification
RUN git clone https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && cd /opt/nccl-tests \
    && make -j $(nproc) MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/opt/nccl/build

#################################################
## ML libraries
RUN pip install \
    transformers>=4.45.0 \
    datasets \
    accelerate \
    tokenizers \
    numpy \
    scipy \
    scikit-learn \
    vllm>=0.7.0 \
    hydra-core \
    omegaconf \
    wandb \
    tensorboard \
    boto3 \
    botocore \
    tenacity \
    langdetect \
    math_verify \
    peft

#################################################
## veRL v0.6.1
WORKDIR /workspace
RUN git clone https://github.com/volcengine/verl.git && \
    cd verl && \
    git checkout v0.6.1
WORKDIR /workspace/verl
RUN pip install -e .

#################################################
## Copy experiment source
COPY src/ /workspace/experiments/src/
WORKDIR /workspace/experiments

# Expose Ray ports
EXPOSE 8265 10001 6379
