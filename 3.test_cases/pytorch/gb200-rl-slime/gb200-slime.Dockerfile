# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# slime RL image for GB200 (arm64 / Grace). NGC Blackwell base; SGLang rollout + TE.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.02-py3   # arm64; CUDA 13 / torch 2.11 era
FROM ${BASE_IMAGE}

ARG SLIME_REF=v0.2.4                 # AWS test case pin (upstream v0.3.0 also works)
ARG SGLANG_VERSION=0.5.12.post1      # unified cu130 tag, SM100 default
ARG TE_VERSION=2.13
ARG EFA_INSTALLER_VERSION=1.48.0
ENV DEBIAN_FRONTEND=noninteractive

RUN curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz -o /tmp/efa.tar.gz && \
    cd /tmp && tar -xf efa.tar.gz && cd aws-efa-installer && \
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify && \
    rm -rf /tmp/efa.tar.gz /tmp/aws-efa-installer
ENV PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH

RUN pip install --no-cache-dir "sglang==${SGLANG_VERSION}" "transformer-engine[pytorch]>=${TE_VERSION}" ray || true
RUN git clone https://github.com/THUDM/slime.git /opt/slime && \
    cd /opt/slime && git checkout ${SLIME_REF} && pip install --no-cache-dir -e . || \
    echo "NOTE: slime build may need Grace-Blackwell NCCL fixes; see README."

COPY run-grpo.sh smoke.sh /opt/slime/
RUN chmod +x /opt/slime/run-grpo.sh /opt/slime/smoke.sh
WORKDIR /opt/slime
