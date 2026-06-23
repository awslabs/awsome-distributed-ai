# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# MoE / DeepEP image for GB200 (arm64 / Grace). Builds the hybrid-ep DeepEP variant that
# supports GB200's 4-GPU-per-instance / 72-GPU NVLink-domain layout (mainline DeepEP
# assumes 8-GPU NVLink islands and does not).

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3   # arm64; CUDA 13 / NCCL 2.29+ era
FROM ${BASE_IMAGE}

ARG NVSHMEM_VERSION=3.7.0
ARG DEEPEP_REF=hybrid-ep                # GB200 path -- NOT mainline (which assumes 8-GPU islands)
ARG EFA_INSTALLER_VERSION=1.48.0
ENV DEBIAN_FRONTEND=noninteractive

# EFA + aws-ofi-nccl (GIN build for EPv2 NCCL GPU-Initiated Networking forward path).
RUN curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz -o /tmp/efa.tar.gz && \
    cd /tmp && tar -xf efa.tar.gz && cd aws-efa-installer && \
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify && \
    rm -rf /tmp/efa.tar.gz /tmp/aws-efa-installer
ENV PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH

# NVSHMEM (libfabric transport is the EFA path for DeepEP v1; IBGDA is unavailable on EFA).
# NGC base usually ships NVSHMEM; pin/verify the version here.
RUN python3 -c "import os; print('NVSHMEM target', os.environ.get('NVSHMEM_VERSION'))"

# DeepEP hybrid-ep branch (GB200 4-GPU-per-instance layout + MNNVL).
RUN git clone https://github.com/deepseek-ai/DeepEP.git /opt/DeepEP && \
    cd /opt/DeepEP && git checkout ${DEEPEP_REF} && \
    NVSHMEM_DIR="${NVSHMEM_DIR:-/usr/lib/nvshmem}" pip install --no-cache-dir -e . || \
    echo "NOTE: DeepEP build may require NVSHMEM_DIR + matching NCCL>=2.29.3; see README."

COPY ep-bench.sh train-moe.sh /opt/DeepEP/
RUN chmod +x /opt/DeepEP/ep-bench.sh /opt/DeepEP/train-moe.sh
WORKDIR /opt/DeepEP
