# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Disaggregated prefill-decode image for GB200 (arm64 / Grace): vLLM + NIXL over EFA.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3   # arm64; CUDA >= 12.8 (sm_100)
FROM ${BASE_IMAGE}

ARG VLLM_VERSION=0.21.0
ARG NIXL_REF=1.1.0
ARG EFA_INSTALLER_VERSION=1.48.0
ENV DEBIAN_FRONTEND=noninteractive

# EFA (provides /opt/amazon/efa for NIXL's libfabric backend).
RUN curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz -o /tmp/efa.tar.gz && \
    cd /tmp && tar -xf efa.tar.gz && cd aws-efa-installer && \
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify && \
    rm -rf /tmp/efa.tar.gz /tmp/aws-efa-installer
ENV PATH=/opt/amazon/efa/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH

# vLLM (Blackwell aarch64 wheel; FA4 default on sm_100/sm_103).
RUN pip install --no-cache-dir "vllm==${VLLM_VERSION}"

# NIXL built against the EFA libfabric (the LIBFABRIC backend is the AWS KV path).
RUN git clone https://github.com/ai-dynamo/nixl.git /opt/nixl && \
    cd /opt/nixl && git checkout ${NIXL_REF} && \
    pip install --no-cache-dir . --config-settings=cmake.args="-Dlibfabric_path=/opt/amazon/efa" || \
    echo "NOTE: NIXL EFA build needs libfabric headers from /opt/amazon/efa; see README + issue #1609."

# Workaround for NIXL #1609 (LIBFABRIC+VRAM dma-buf 'Bad address' on p6e-gb200).
ENV FI_HMEM_CUDA_USE_DMABUF=0

COPY nixl-config.yaml /opt/disagg/nixl-config.yaml
WORKDIR /opt/disagg
