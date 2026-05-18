# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# NeMo RL Workshop — Multi-arch (g5/g6e/p5) + EFA
#
# Minified Dockerfile for the resilient-training workshop.
# Builds on CPU (no GPU required) and supports:
#   - g5.8xlarge   (A10G,  SM 8.6, Ampere)
#   - g6e.48xlarge (L40S,  SM 8.9, Ada Lovelace)
#   - p5.48xlarge  (H100,  SM 9.0, Hopper)
#
# What's included:
#   - NeMo RL core (torch 2.9, ray 2.49, transformers 4.57)
#   - vLLM 0.11 (inference / generation)
#   - DTensor + LoRA (automodel: flash-attn, transformer-engine)
#   - EFA networking (latest libfabric + aws-ofi-nccl 1.18.0)
#   - nvidia-resiliency-ext (NVRx) for fault tolerance
#
# What's NOT included (saves ~20 min build time + ~4GB image):
#   - sglang (alternative inference backend, not used)
#   - mcore / Megatron-LM (uses DTensor instead)
#   - nsight-systems-cli (profiler, not needed for workshop)
#   - dev/test/docs dependency groups
#
# Build (multi-arch, ~40 min on c5.4xlarge):
#   docker build -t nemo-rl-workshop .
#
# Build (single-arch, ~25 min — pick one for fast iteration):
#   docker build --build-arg TORCH_CUDA_ARCH_LIST="8.6" -t nemo-rl-workshop:g5 .
#
# CUDA arch note:
#   flash-attn and transformer-engine compile for all target archs.
#   deep_ep and deep_gemm (DeepSeek) use Hopper TMA instructions and
#   compile for SM 9.0 only — they're not called at runtime on g5/g6e.
#
# Build args:
#   TORCH_CUDA_ARCH_LIST  Target GPU archs (default: "8.6 8.9 9.0")
#   NRL_GIT_REF           NeMo RL commit/branch (default: pinned)
#   MAX_JOBS              Limit parallel compilation (prevent OOM)
#   OFI_NCCL_VERSION      aws-ofi-nccl version (default: v1.18.0)
#   EFA_INSTALLER_VERSION AWS EFA installer version (default: 1.47.0)

# ============================================================
# Stage 1: Clone NeMo RL source
# ============================================================
ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:25.05-cuda12.9-devel-ubuntu24.04
FROM scratch AS nemo-rl-src
ARG NRL_GIT_REF=c40dba37789c64d03fcce72b152e5657ae6aacd7
ADD --keep-git-dir=true https://github.com/NVIDIA-NeMo/RL.git#${NRL_GIT_REF} /

# ============================================================
# Stage 2: System dependencies + Python toolchain
# ============================================================
FROM ${BASE_IMAGE} AS base
ENV NRL_CONTAINER=1
USER root

RUN <<"EOF" bash -exu -o pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    jq curl git rsync wget less vim
apt-get clean
rm -rf /var/lib/apt/lists/*
EOF

# CMake (needed for sgl-kernel/flashinfer cmake builds, keeping for vllm)
RUN ARCH=$(uname -m) \
    && CMAKE_VERSION=3.31.1 \
    && CMAKE_INSTALLER="cmake-${CMAKE_VERSION}-linux-${ARCH}" \
    && curl --retry 3 --retry-delay 2 -fsSL -o "${CMAKE_INSTALLER}.tar.gz" \
        "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_INSTALLER}.tar.gz" \
    && tar -xzf "${CMAKE_INSTALLER}.tar.gz" \
    && cp -r "${CMAKE_INSTALLER}/bin/"* /usr/local/bin/ \
    && cp -r "${CMAKE_INSTALLER}/share/"* /usr/local/share/ \
    && rm -rf "${CMAKE_INSTALLER}" "${CMAKE_INSTALLER}.tar.gz"

ARG UV_VERSION=0.9.7
ARG PYTHON_VERSION=3.12
ENV PATH="/root/.local/bin:$PATH"
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh && \
    uv python install ${PYTHON_VERSION}

ENV RAY_USAGE_STATS_ENABLED=0
ENV RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
ENV NEMO_RL_VENV_DIR=/opt/ray_venvs

# ============================================================
# Stage 3: Python dependencies (vllm + automodel only)
# ============================================================
FROM base AS hermetic
WORKDIR /opt/nemo-rl

ARG MAX_JOBS
ARG NVTE_BUILD_THREADS_PER_JOB

ENV UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
ENV UV_LINK_MODE=copy

# Multi-arch: g5 (A10G 8.6) + g6e (L40S 8.9) + p5 (H100 9.0)
# Override with --build-arg TORCH_CUDA_ARCH_LIST="8.6" for fast single-arch builds
ARG TORCH_CUDA_ARCH_LIST="8.6 8.9 9.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Copy only what uv needs to resolve dependencies
COPY --from=nemo-rl-src pyproject.toml uv.lock ./
COPY --from=nemo-rl-src nemo_rl/__init__.py nemo_rl/package_info.py ./nemo_rl/
COPY --from=nemo-rl-src tools/build-custom-vllm.sh ./tools/build-custom-vllm.sh
COPY --from=nemo-rl-src tools/build-custom-flashinfer.sh ./tools/build-custom-flashinfer.sh
COPY --from=nemo-rl-src --link research/ ./research/
COPY --from=nemo-rl-src --link 3rdparty/ ./3rdparty/

RUN <<"EOF" bash -exu
uv venv --seed

# Core dependencies (torch, ray, transformers, etc.) — no CUDA compilation
uv sync --locked --no-install-project

# vLLM extra — deep_ep and deep_gemm use Hopper TMA instructions (cp.async.bulk,
# elect) that only exist on SM 9.0+. They CANNOT compile for 8.6/8.9.
# Build them targeting SM 9.0 only; they're not called at runtime on g5/g6e anyway.
TORCH_CUDA_ARCH_LIST="9.0" uv sync --locked --extra vllm --no-install-project

# automodel extra — flash-attn and transformer-engine compile for all target archs.
# deep_ep is already installed from the vllm step above, so uv skips rebuilding it.
uv sync --locked --extra automodel --no-install-project

# NOTE: sglang and mcore are intentionally skipped.
# Add them back if needed:
#   uv sync --locked --extra sglang --no-install-project
#   uv sync --locked --extra mcore --no-install-project

# Clean build caches (saves 4-6GB)
rm -rf /root/.cache/uv
# Clean Ray aiohttp duplicate
find /opt/nemo_rl_venv -type d -name "aiohttp" -path "*/ray/_private/*" -exec rm -rf {} + 2>/dev/null || true
EOF

ENV PATH="/opt/nemo_rl_venv/bin:$PATH"
ENV NEMO_RL_VENV_DIR=/opt/ray_venvs
WORKDIR /opt/nemo-rl

# ============================================================
# Stage 4: Full repo + Ray worker venv prefetch
# ============================================================
FROM hermetic AS release

ENV NEMO_RL_VENV_DIR=/opt/ray_venvs

COPY --from=nemo-rl-src . /opt/nemo-rl
RUN git rev-parse --is-shallow-repository | grep -q true && git fetch --unshallow || true

RUN <<"EOF" bash -exu
# Prefetch Ray worker venvs at build time (saves ~2 min at first launch).
# Skip sglang and mcore backends since they're not installed.
uv run nemo_rl/utils/prefetch_venvs.py --negative-filters sglang mcore

# Clean build caches
rm -rf /root/.cache/uv
EOF

RUN python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint

# SBOM / OSS compliance — generates /THIRD-PARTY-LICENSES and /SBOM.txt
COPY docker/generate_sbom.sh /tmp/generate_sbom.sh
RUN bash /tmp/generate_sbom.sh && rm /tmp/generate_sbom.sh

# ============================================================
# Stage 5: EFA networking
# ============================================================
FROM release AS efa

LABEL maintainer="antonai"
LABEL description="NeMo RL workshop (multi-arch g5/g6e/p5) + EFA"

# Build deps for EFA installer + aws-ofi-nccl
RUN apt-get update && apt-get install -y --no-install-recommends \
        autoconf automake libtool libhwloc-dev \
        pciutils environment-modules tcl \
        udev iproute2 ethtool dmidecode \
        libevent-core-2.1-7t64 libevent-pthreads-2.1-7t64 libevent-dev \
    && rm -rf /var/lib/apt/lists/*

# EFA installer — pinned to a specific version per repo convention (CI requires >= 1.47.0)
# Minimal mode: libfabric + rdma-core only, no OpenMPI.
ARG EFA_INSTALLER_VERSION=1.47.0
RUN cd /tmp && \
    curl -sL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
        | tar xz && \
    cd aws-efa-installer && \
    ./efa_installer.sh -y --minimal --skip-kmod --skip-limit-conf --no-verify && \
    cd / && rm -rf /tmp/aws-efa-installer

# aws-ofi-nccl (NCCL <-> EFA bridge)
ARG OFI_NCCL_VERSION=v1.18.0
ARG OFI_NCCL_PREFIX=/opt/aws-ofi-nccl

RUN git clone --depth 1 --branch ${OFI_NCCL_VERSION} \
        https://github.com/aws/aws-ofi-nccl.git /tmp/aws-ofi-nccl && \
    cd /tmp/aws-ofi-nccl && \
    ./autogen.sh && \
    ./configure \
        --prefix=${OFI_NCCL_PREFIX} \
        --with-libfabric=/opt/amazon/efa \
        --with-cuda=/usr/local/cuda \
        --with-nccl=/usr && \
    make -j"$(nproc)" && \
    make install && \
    rm -rf /tmp/aws-ofi-nccl && \
    apt-get purge -y --auto-remove autoconf automake libtool && \
    rm -rf /var/lib/apt/lists/*

# Runtime library paths
ENV LD_LIBRARY_PATH="/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# EFA / NCCL defaults
# NOTE: g5 (EFA Gen1) needs NCCL_NET_PLUGIN=none at runtime.
#       g6e/p5 (EFA Gen2+) use ofi. Set in your K8s manifest per instance type.
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1
ENV NCCL_NET_PLUGIN=ofi
ENV NCCL_SOCKET_IFNAME="^lo,docker,veth,eni"

# Repo-level license files
COPY THIRD-PARTY-LICENSES UTILITY-LICENSES /opt/nemo-rl/

WORKDIR /opt/nemo-rl
