# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# FP8/FP4 Megatron-LM training image for GB200 (arm64 / Grace).
# Extends an NGC PyTorch base that already ships a Blackwell-ready TE + CUDA toolchain.

# arm64/sbsa NGC PyTorch, Blackwell-ready (TE >= 2.16, CUDA 12.8+). Pin a tested tag.
# Do NOT use an x86_64 base -- GB200 is Grace (ARM). That would be the B200/B300 path.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3
FROM ${BASE_IMAGE}

ARG MEGATRON_LM_REF=core_r0.17.0
ARG EFA_INSTALLER_VERSION=1.48.0

ENV DEBIAN_FRONTEND=noninteractive

# EFA + bundled aws-ofi-nccl + OpenMPI (runtime collectives over EFA cross-UltraServer).
RUN curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz -o /tmp/efa.tar.gz && \
    cd /tmp && tar -xf efa.tar.gz && cd aws-efa-installer && \
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify && \
    rm -rf /tmp/efa.tar.gz /tmp/aws-efa-installer
ENV PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH

# Megatron-LM (TE comes from the NGC base; do not rebuild it).
RUN git clone https://github.com/NVIDIA/Megatron-LM.git /opt/Megatron-LM && \
    cd /opt/Megatron-LM && git checkout ${MEGATRON_LM_REF} && \
    pip install --no-cache-dir -e .

COPY train.sh /opt/Megatron-LM/train.sh
RUN chmod +x /opt/Megatron-LM/train.sh
WORKDIR /opt/Megatron-LM
