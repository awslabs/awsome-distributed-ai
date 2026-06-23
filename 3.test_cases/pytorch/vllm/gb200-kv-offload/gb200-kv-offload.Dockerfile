# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# KV-offload-to-Grace serving image for GB200 (arm64 / Grace).
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3   # arm64; sm_100
FROM ${BASE_IMAGE}

ARG VLLM_VERSION=0.14.0
ARG LMCACHE_VERSION=0.4.7
ENV DEBIAN_FRONTEND=noninteractive

RUN pip install --no-cache-dir "vllm==${VLLM_VERSION}" "lmcache==${LMCACHE_VERSION}" || \
    pip install --no-cache-dir "vllm==${VLLM_VERSION}"

COPY serve.sh /opt/kv-offload/serve.sh
RUN chmod +x /opt/kv-offload/serve.sh
WORKDIR /opt/kv-offload
# Grace LPDDR5X (~480 GB, ~1/16 HBM bandwidth) is a COLD KV tier; active KV stays in HBM.
