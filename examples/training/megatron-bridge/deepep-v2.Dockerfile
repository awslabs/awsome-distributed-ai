# syntax=docker/dockerfile:1.7
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Opt-in image. Build from the repository root with 1.build-deepep-v2.sh.
# The NeMo release base supplies Python 3.12, CUDA 13.3, Torch and TE.
ARG NEMO_IMAGE=nvcr.io/nvidia/nemo:26.08
ARG TORCH_CUDA_ARCH_LIST="9.0;10.0;10.3"
FROM ${NEMO_IMAGE} AS transport
ARG EFA_INSTALLER_VERSION=1.50.0
ARG GDRCOPY_VERSION=v2.5.2
ARG NCCL_VERSION=2.31.2
# Commit by necessity: amazon-contributing/DeepEP publishes no releases or tags.
ARG DEEPEP_COMMIT=874779c9ccd2294b56304bd6cc5f138f1f71d097
ARG MAX_JOBS=4
ARG TORCH_CUDA_ARCH_LIST
ENV PATH=/opt/venv/bin:/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH
# Same userspace EFA/GDRCopy recipe as deepep-v2-benchmark/deepep.Dockerfile.
# Retain the NeMo base's MPI/verbs libraries: its Torch depends on them.
RUN apt-get update && apt-get install -y --no-install-recommends \
      autoconf automake build-essential cmake curl git libtool ninja-build && \
    git clone --branch ${GDRCOPY_VERSION} --depth 1 https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy && \
    make -j${MAX_JOBS} -C /tmp/gdrcopy prefix=/opt/gdrcopy lib lib_install && rm -rf /tmp/gdrcopy && \
    curl --retry 3 --retry-delay 2 -fsSL \
      https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
      -o /tmp/efa.tar.gz && \
    tar -xf /tmp/efa.tar.gz -C /tmp && \
    cd /tmp/aws-efa-installer && \
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify && \
    cd / && rm -rf /tmp/aws-efa-installer /tmp/efa.tar.gz /var/lib/apt/lists/*
RUN python3 -m pip install --no-deps nvidia-nccl-cu13==${NCCL_VERSION} && \
    mkdir -p /opt/nccl && \
    ln -s "$(python3 -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.nccl").submodule_search_locations)))')" /opt/nccl/current && \
    printf '#!/bin/sh\nexec /opt/venv/bin/python3 -m pip "$@"\n' > /opt/deepep-pip && chmod +x /opt/deepep-pip
ENV LD_LIBRARY_PATH=/opt/nccl/current/lib:/opt/gdrcopy/lib:/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib:$LD_LIBRARY_PATH
# Use the merged installer directly, without vendoring or transport patches.
COPY micro-benchmarks/expert-parallelism/deepep-v2-benchmark/setup_deepep_gin.sh /opt/setup_deepep_gin.sh
RUN CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS} MAKEFLAGS=-j${MAX_JOBS} \
      bash /opt/setup_deepep_gin.sh --deepep-ref ${DEEPEP_COMMIT} \
      --nccl-root /opt/nccl/current --python /opt/venv/bin/python3 --pip /opt/deepep-pip && \
    test "$(git -C /opt/amazon/deepep rev-parse HEAD)" = "${DEEPEP_COMMIT}"

FROM transport AS upstream
# Release tags are used wherever a usable release exists (NeMo, EFA, GDRCopy,
# NCCL above). These two stay commits by necessity: the deepepv2 flex-dispatcher
# backend exists in no released megatron-core (checked core_v0.19.0) and the
# PR #5153 init backport applies to this exact dev head; Bridge 0.7.0 is
# unreleased and no released Bridge is paired with that unreleased Core. The
# baked verifier asserts these exact sources, so they are fixed, not tunable.
ARG MCORE_COMMIT=bb5dfd08f09ce06c5925af453fef06b3129f199d
ARG BRIDGE_COMMIT=281f4bebfd78eb7cc9e8282c7929e99a9746b6cc
# Install the pinned sources. The base supplies the training dependencies;
# --no-deps prevents a resolver from replacing Torch, NCCL or our Core pin.
RUN git clone --filter=blob:none https://github.com/NVIDIA/Megatron-LM.git /opt/upstream/Megatron-LM && \
    git -C /opt/upstream/Megatron-LM checkout --detach ${MCORE_COMMIT} && \
    git clone --filter=blob:none https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /opt/upstream/Megatron-Bridge && \
    git -C /opt/upstream/Megatron-Bridge checkout --detach ${BRIDGE_COMMIT} && \
    python3 -m pip install --no-deps --no-build-isolation -e /opt/upstream/Megatron-LM -e /opt/upstream/Megatron-Bridge
# FI_*/PYTHONPATH come from the launcher and the editable installs; the image
# only pins what selects the EFA plugin and the DeepEP v2 GIN backend, plus the
# supported GPU architectures (kept at runtime so the base image's broader,
# sm_103-less list never leaks into a runtime extension build).
ARG TORCH_CUDA_ARCH_LIST
ENV MEGATRON_VARIANT=dev-deepepv2 MOE_DISPATCHER=deepepv2 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    NCCL_NET_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so \
    NCCL_GIN_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so \
    NCCL_GIN_TYPE=5 NCCL_SYM_GIN_KERNELS_ENABLE=0 NVIDIA_GDRCOPY=enabled
COPY examples/training/megatron-bridge/mcore-pr5153-comm-init.patch /opt/benchmark/mcore-pr5153-comm-init.patch
# Only the same-subgroup initialization hunk from PR #5153's pinned head.
# The verifier requires this exact patch and rejects any other Core/Bridge diff.
RUN git -C /opt/upstream/Megatron-LM apply --check /opt/benchmark/mcore-pr5153-comm-init.patch && \
    git -C /opt/upstream/Megatron-LM apply /opt/benchmark/mcore-pr5153-comm-init.patch
COPY examples/training/megatron-bridge/kimi-k2/benchmarks/bench_kimi_k2_pretrain.py /opt/benchmark/bench_kimi_k2_pretrain.py
COPY examples/training/megatron-bridge/2.verify-deepep-v2.py /opt/benchmark/verify_deepep_v2.py
RUN python3 /opt/benchmark/verify_deepep_v2.py --cuda-stub
WORKDIR /workspace
