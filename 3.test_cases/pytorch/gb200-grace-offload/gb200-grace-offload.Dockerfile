# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Grace unified-memory offload image for GB200 (arm64 / Grace).
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3   # arm64; CUDA 12.x unified memory + ATS
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
# DeepSpeed >= 0.18.0 for SuperOffload (Grace-aware coherent-memory offload).
RUN pip install --no-cache-dir "deepspeed>=0.18.0"

COPY run.sh ds_config_superoffload.json /opt/grace-offload/
RUN chmod +x /opt/grace-offload/run.sh
WORKDIR /opt/grace-offload
# Note: MPAM/resctrl bandwidth partitioning needs a privileged container or host config.
