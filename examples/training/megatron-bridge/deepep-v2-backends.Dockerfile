# syntax=docker/dockerfile:1.7
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Backend-only derivatives; Core, Bridge, Torch, CUDA, TE, NCCL and EFA stay pinned.
ARG COMMON_IMAGE
FROM ${COMMON_IMAGE} AS backend-base
RUN python3 -m pip uninstall -y deep-ep && \
    python3 /opt/benchmark/verify_deepep_v2.py --source-only

FROM backend-base AS uccl
ARG MAX_JOBS=4
ARG UCCL_COMMIT=0dc87eb3b40c372a16b70ef320f37daaa5299ca7
ARG NANOBIND_VERSION=2.4.0
# Same UCCL source and explicit architectures as the merged training recipe.
RUN apt-get update && apt-get install -y --no-install-recommends libnuma-dev libnl-3-dev libnl-route-3-dev libibverbs-dev && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --no-deps nanobind==${NANOBIND_VERSION} && \
    git clone --filter=blob:none https://github.com/uccl-project/uccl.git /opt/uccl && \
    git -C /opt/uccl checkout --detach ${UCCL_COMMIT} && \
    git -C /opt/uccl submodule update --init --recursive && \
    MAKEFLAGS=-j${MAX_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS} \
      python3 -m pip install --no-deps --no-build-isolation /opt/uccl
RUN cd /opt/uccl/ep && \
    MAX_JOBS=${MAX_JOBS} PER_EXPERT_BATCHING=1 TORCH_CUDA_ARCH_LIST="10.0a+PTX;10.3a+PTX" python3 setup.py install && \
    python3 -m pip install --no-deps --no-build-isolation /opt/uccl/ep/deep_ep_wrapper
# PER_EXPERT_BATCHING stays visible at runtime: the kernels were compiled with
# it and the measurement receipts record it as arm identity.
ENV MOE_DISPATCHER=deepep PER_EXPERT_BATCHING=1
RUN python3 /opt/benchmark/verify_deepep_v2.py --source-only && \
    python3 -c "import ctypes; ctypes.CDLL('/usr/local/cuda/lib64/stubs/libcuda.so',mode=ctypes.RTLD_GLOBAL); import deep_ep,uccl.ep; assert hasattr(deep_ep,'Buffer') and not hasattr(deep_ep,'ElasticBuffer'); assert '/opt/venv/' in deep_ep.__file__; print(deep_ep.__file__,uccl.ep.__file__)"

FROM backend-base AS deepep-v1
ARG MAX_JOBS=4
ARG NVSHMEM_COMMIT=d72cf233da003ad3b5a30188f9902886b5677da9
ARG DEEPEP_V1_COMMIT=567632dd59810d77b3cc05553df953cc0f779799
COPY examples/training/megatron-bridge/deepep/setup_deepep_efa.sh /opt/benchmark/setup_deepep_v1_efa.sh
# Reuse the merged EFA shim verbatim; only bound its build's make concurrency.
RUN git clone --filter=blob:none https://github.com/NVIDIA/nvshmem.git /opt/nvshmem_src && \
    git -C /opt/nvshmem_src checkout --detach ${NVSHMEM_COMMIT} && \
    git clone --filter=blob:none https://github.com/deepseek-ai/DeepEP.git /opt/deepep-v1-src && \
    git -C /opt/deepep-v1-src checkout --detach ${DEEPEP_V1_COMMIT} && \
    python3 -m pip uninstall -y nvidia-nvshmem-cu13 nvidia-nvshmem-cu12 && \
    sed -i 's/make -j"$(nproc)"/make -j"${MAX_JOBS}"/' /opt/benchmark/setup_deepep_v1_efa.sh && \
    bash /opt/benchmark/setup_deepep_v1_efa.sh --venv /opt/venv \
      --cuda-home /usr/local/cuda --libfabric-home /opt/amazon/efa \
      --gdrcopy-home /opt/gdrcopy --gpu-arch 100 \
      --deepep-src /opt/deepep-v1-src --nvshmem-src /opt/nvshmem_src
ENV MOE_DISPATCHER=deepep \
    NVSHMEM_DIR=/opt/nvshmem_src/install NVSHMEM_HOME=/opt/nvshmem_src/install \
    NVSHMEM_REMOTE_TRANSPORT=libfabric NVSHMEM_LIBFABRIC_PROVIDER=efa \
    LD_LIBRARY_PATH=/opt/nvshmem_src/install/lib:$LD_LIBRARY_PATH
RUN python3 /opt/benchmark/verify_deepep_v2.py --source-only && \
    python3 -c "import ctypes; ctypes.CDLL('/usr/local/cuda/lib64/stubs/libcuda.so',mode=ctypes.RTLD_GLOBAL); import deep_ep; assert hasattr(deep_ep,'Buffer') and not hasattr(deep_ep,'ElasticBuffer'); assert '/opt/venv/' in deep_ep.__file__; print(deep_ep.__file__)"
