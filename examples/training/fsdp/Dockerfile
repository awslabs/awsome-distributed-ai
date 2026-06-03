# Pinned base image. The companion PR in micro-benchmarks/nccl-tests/ rebuilds
# this image with CUDA 13, NCCL v2.30.4, EFA installer 1.48, and `sm_103` in
# the gencode line for native B300 (Blackwell Ultra) collectives. Refresh this
# tag whenever you re-cut nccl-tests; do not pin to `:latest`.
FROM public.ecr.aws/hpc-cloud/nccl-tests:cuda13.0.2-efa1.48.0-ofiv1.19.0-ncclv2.30.4-1-testsv2.18.3 AS base

RUN apt update && apt install -y nvtop

RUN mkdir -p /fsdp /checkpoints
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY src/ /fsdp/
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /fsdp/requirements.txt

WORKDIR /fsdp

# HyperPod-specific image with elastic agent
FROM base AS hpto

RUN pip install hyperpod-elastic-agent==1.1.2
