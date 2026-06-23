# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# FP4 (NVFP4) TensorRT-LLM serving image for GB200 (arm64 / Grace, sm_100).
# Built FROM the official TensorRT-LLM release container (arm64 variant).
ARG TRTLLM_VERSION=1.2.0
FROM nvcr.io/nvidia/tensorrt-llm/release:${TRTLLM_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
# NVIDIA ModelOpt for the BYO NVFP4 quantization path + lm-eval for the accuracy gate.
RUN pip install --no-cache-dir "nvidia-modelopt[torch]>=0.23,<0.45" lm-eval || true

COPY quantize.sh serve.sh bench.sh accuracy_gate.sh /opt/fp4/
RUN chmod +x /opt/fp4/*.sh
WORKDIR /opt/fp4
# sm_100 = GB200 (Blackwell). Do NOT assume an x86 host -- this is Grace (arm64).
