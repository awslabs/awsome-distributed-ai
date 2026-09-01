# First stage - DLC PyTorch 2.6, Python 3.12, CUDA 12.6, Ubuntu 22.04
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-ec2 AS dlc

# Second stage - Slurm compute node
# NOTE: Ubuntu version mismatch (22.04 in Stage 1 vs 24.04 here) is intentional.
# The Slinky project does not publish slurmd images for ubuntu22.04, and AWS DLC
# does not publish GPU training images for ubuntu24.04. The copied libraries
# (CUDA, EFA, NCCL, OpenMPI, Python) are dynamically linked against glibc, but
# glibc is backward-compatible: binaries built against 22.04's glibc 2.35 run on
# 24.04's glibc 2.39. The reverse direction (newer -> older) would break.
FROM ghcr.io/slinkyproject/slurmd:25.11.1-ubuntu24.04

ARG PYTHON_SHORT_VERSION=3.12

# Create required directory
RUN mkdir -p /var/spool/slurmd

# Environment variables from DLC
ENV CUDA_HOME="/usr/local/cuda" \
    EFA_PATH="/opt/amazon/efa" \
    OPEN_MPI_PATH="/opt/amazon/openmpi"

ENV LD_LIBRARY_PATH="lib:${EFA_PATH}/lib:${OPEN_MPI_PATH}/lib:${CUDA_HOME}/lib64:/usr/local/lib:/lib/x86_64-linux-gnu:/opt/nccl/build/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/usr/local/nvidia/lib" \
    PATH="${EFA_PATH}/bin:${OPEN_MPI_PATH}/bin:${CUDA_HOME}/bin:${PATH}" \
    NCCL_DEBUG=INFO \
    NCCL_SOCKET_IFNAME=^docker0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    NVTE_FRAMEWORK=pytorch

# Install critical system dependencies missing in base Slurm image
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libcurl4-openssl-dev \
    libssl-dev \
    libnuma1 \
    libnuma-dev \
    libibverbs-dev \
    libtool \
    autoconf \
    pkg-config \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy CUDA stack from DLC
COPY --from=dlc /usr/local/cuda /usr/local/cuda

# Copy EFA stack from DLC
COPY --from=dlc /opt/amazon/efa /opt/amazon/efa
COPY --from=dlc /opt/amazon/openmpi /opt/amazon/openmpi
COPY --from=dlc /opt/amazon/ofi-nccl /opt/amazon/ofi-nccl

# Copy NCCL configuration
COPY --from=dlc /usr/local/lib/libnccl* /usr/local/lib/
COPY --from=dlc /etc/nccl.conf /etc/nccl.conf

# Configure OpenMPI
RUN mv ${OPEN_MPI_PATH}/bin/mpirun ${OPEN_MPI_PATH}/bin/mpirun.real \
    && echo '#!/bin/bash' > ${OPEN_MPI_PATH}/bin/mpirun \
    && echo "${OPEN_MPI_PATH}/bin/mpirun.real --allow-run-as-root \"\$@\"" >> ${OPEN_MPI_PATH}/bin/mpirun \
    && chmod a+x ${OPEN_MPI_PATH}/bin/mpirun \
    && echo "hwloc_base_binding_policy = none" >> ${OPEN_MPI_PATH}/etc/openmpi-mca-params.conf \
    && echo "rmaps_base_mapping_policy = slot" >> ${OPEN_MPI_PATH}/etc/openmpi-mca-params.conf

# Copy Python installation
COPY --from=dlc /usr/local/bin/python${PYTHON_SHORT_VERSION}* /usr/local/bin/
COPY --from=dlc /usr/local/lib/python${PYTHON_SHORT_VERSION} /usr/local/lib/python${PYTHON_SHORT_VERSION}
COPY --from=dlc /usr/local/lib/libpython${PYTHON_SHORT_VERSION}* /usr/local/lib/
COPY --from=dlc /usr/local/include/python${PYTHON_SHORT_VERSION}* /usr/local/include/

# Fix Python symlinks
RUN rm -f /usr/local/bin/python3 && \
    rm -f /usr/local/bin/python && \
    ln -s /usr/local/bin/python${PYTHON_SHORT_VERSION} /usr/local/bin/python3 && \
    ln -s /usr/local/bin/python${PYTHON_SHORT_VERSION} /usr/local/bin/python

# Additional requirements
RUN /usr/local/bin/python3 -m pip install --no-cache-dir \
    transformers==4.37.2 \
    datasets==2.17.1

# Remove problematic typing.py to avoid conflicts
RUN rm -f /usr/local/lib/python${PYTHON_SHORT_VERSION}/site-packages/typing.py

# Install OpenSSH, allow OpenSSH to talk to containers without asking for confirmation
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-client openssh-server \
    && mkdir -p /var/run/sshd \
    && cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new \
    && echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new \
    && mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Configure OpenSSH so that nodes can communicate with each other
RUN mkdir -p /var/run/sshd \
    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN rm -rf /root/.ssh/ \
    && mkdir -p /root/.ssh/ \
    && ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa \
    && cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys \
    && printf "Host *\n StrictHostKeyChecking no\n" >> /root/.ssh/config

WORKDIR /home
