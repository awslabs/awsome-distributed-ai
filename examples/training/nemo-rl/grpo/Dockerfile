# syntax=docker/dockerfile:1.10
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# NeMo RL Training Image — Multi-Architecture Buildx Pattern
#
# Production-ready Dockerfile for NeMo RL GRPO training with:
#   - Proper --platform ARG handling for docker buildx
#   - Multi-stage build for optimal layer caching
#   - SBOM generation at build time
#   - EFA networking stack (libfabric 2.x + aws-ofi-nccl)
#   - H200/NVSwitch NCCL workarounds baked in
#
# Architecture: linux/amd64 (GPU workloads require x86_64)
#
# This Dockerfile uses the buildx multi-stage pattern where TARGETPLATFORM
# and TARGETARCH are automatically injected by BuildKit. GPU images are
# amd64-only, but the pattern is forward-compatible with arm64 utility
# builds (e.g., data preprocessing on Graviton).
#
# Target GPU configurations (select via TORCH_CUDA_ARCH_LIST):
#   P5en H200 (SM 9.0):  --build-arg TORCH_CUDA_ARCH_LIST="9.0"
#   P5 H100 (SM 9.0):    --build-arg TORCH_CUDA_ARCH_LIST="9.0"
#   Multi-GPU:            --build-arg TORCH_CUDA_ARCH_LIST="8.6 8.9 9.0"
#
# Build:
#   docker buildx build --platform linux/amd64 \
#     -f Dockerfile \
#     -t nemo-rl:latest .
#
# Build with push:
#   docker buildx build --platform linux/amd64 \
#     -f Dockerfile \
#     -t <your-registry>/nemo-rl-workshop:latest \
#     --push .
#
# Build args:
#   BASE_IMAGE              CUDA base image (default: cuda-dl-base 25.05)
#   NRL_GIT_REF             NeMo RL commit/branch to build from
#   TORCH_CUDA_ARCH_LIST    Target GPU SM versions (default: "9.0")
#   UV_VERSION              uv package manager version
#   PYTHON_VERSION          Python version (default: 3.12)
#   OFI_NCCL_VERSION        aws-ofi-nccl version (default: v1.18.0)
#   EFA_INSTALLER_VERSION   AWS EFA installer version (default: 1.47.0)
#   GDRCOPY_VERSION         GDRCopy version tag (default: v2.5.1)
#   MAX_JOBS                Parallel compilation limit (prevent OOM)
#   EFA_INSTALL_MODE        "full" or "minimal" (default: full for P5en RDMA)

# ============================================================
# Stage 1: Clone NeMo RL source (platform-independent)
# ============================================================
# Global ARGs must be before any FROM to be visible across stages
ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04
ARG NRL_GIT_REF=c40dba37789c64d03fcce72b152e5657ae6aacd7

# Using scratch + ADD to clone the repo avoids needing git in the
# base image during the build. The --keep-git-dir flag preserves
# commit history for fingerprint generation.

FROM scratch AS nemo-rl-src
ARG NRL_GIT_REF
ADD --keep-git-dir=true https://github.com/NVIDIA-NeMo/RL.git#${NRL_GIT_REF} /


# ============================================================
# Stage 2: System dependencies + Python toolchain
# ============================================================
ARG BASE_IMAGE
FROM --platform=linux/amd64 ${BASE_IMAGE} AS base

ARG TARGETARCH

ENV NRL_CONTAINER=1
USER root

# System packages — single layer for cache efficiency
RUN <<"EOF" bash -exu -o pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    jq curl git rsync wget less vim \
    pciutils iproute2 ethtool
apt-get clean
rm -rf /var/lib/apt/lists/*
EOF

# CMake — architecture-aware download
RUN ARCH=$(uname -m) \
    && CMAKE_VERSION=3.31.1 \
    && CMAKE_INSTALLER="cmake-${CMAKE_VERSION}-linux-${ARCH}" \
    && curl --retry 3 --retry-delay 2 -fsSL -o "${CMAKE_INSTALLER}.tar.gz" \
        "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_INSTALLER}.tar.gz" \
    && tar -xzf "${CMAKE_INSTALLER}.tar.gz" \
    && cp -r "${CMAKE_INSTALLER}/bin/"* /usr/local/bin/ \
    && cp -r "${CMAKE_INSTALLER}/share/"* /usr/local/share/ \
    && rm -rf "${CMAKE_INSTALLER}" "${CMAKE_INSTALLER}.tar.gz"

# uv package manager
ARG UV_VERSION=0.9.7
ARG PYTHON_VERSION=3.12
ENV PATH="/root/.local/bin:$PATH"
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh && \
    uv python install ${PYTHON_VERSION}

# Runtime environment
ENV RAY_USAGE_STATS_ENABLED=0
ENV RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
ENV NEMO_RL_VENV_DIR=/opt/ray_venvs


# ============================================================
# Stage 3: Python dependencies
# ============================================================
# This is the most expensive stage (~20 min). Layer ordering is
# optimized so that changing NRL_GIT_REF (full repo copy) does
# NOT invalidate the dependency cache.
#
# Order of operations:
#   1. Copy only pyproject.toml + uv.lock (cache key)
#   2. Install all Python deps (cached if lock unchanged)
#   3. Copy full source (invalidates only later stages)

FROM base AS hermetic
WORKDIR /opt/nemo-rl

ARG MAX_JOBS
ARG NVTE_BUILD_THREADS_PER_JOB
ARG TORCH_CUDA_ARCH_LIST="9.0"

ENV UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
ENV UV_LINK_MODE=copy
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Copy only dependency manifests first (layer cache key)
COPY --from=nemo-rl-src pyproject.toml uv.lock ./
COPY --from=nemo-rl-src nemo_rl/__init__.py nemo_rl/package_info.py ./nemo_rl/
COPY --from=nemo-rl-src tools/build-custom-vllm.sh ./tools/build-custom-vllm.sh
COPY --from=nemo-rl-src tools/build-custom-flashinfer.sh ./tools/build-custom-flashinfer.sh
COPY --from=nemo-rl-src --link research/ ./research/
COPY --from=nemo-rl-src --link 3rdparty/ ./3rdparty/

# Install Python dependencies in dependency order:
# 1. Core (torch, ray, transformers) — no CUDA compilation
# 2. vLLM extra (deep_ep, deep_gemm need SM 9.0 TMA instructions)
# 3. Automodel extra (flash-attn, transformer-engine)
RUN <<"EOF" bash -exu
uv venv --seed

# Core dependencies — no CUDA compilation needed
uv sync --locked --no-install-project

# vLLM — deep_ep and deep_gemm use Hopper-only TMA instructions
# Force SM 9.0 regardless of TORCH_CUDA_ARCH_LIST for these
TORCH_CUDA_ARCH_LIST="9.0" uv sync --locked --extra vllm --no-install-project

# Automodel — flash-attn and transformer-engine compile for target archs
uv sync --locked --extra automodel --no-install-project

# peft — needed by eval script for LoRA adapter merging
uv pip install --python ${UV_PROJECT_ENVIRONMENT}/bin/python "peft>=0.13,<1.0"

# CVE remediation: upgrade vulnerable Python packages to fixed versions
uv pip install --python ${UV_PROJECT_ENVIRONMENT}/bin/python \
    "h11>=0.16.0" \
    "Pillow>=10.3.0" \
    "certifi>=2024.7.4" \
    "urllib3>=2.2.2" \
    "requests>=2.32.0" \
    "idna>=3.7" \
    "jinja2>=3.1.5" \
    "aiohttp>=3.13.3" \
    "tornado>=6.4.2" \
    "setuptools>=75.6.0" \
    "cryptography>=46.0.5" \
    "zipp>=3.19.1" \
    "tqdm>=4.66.4" \
    "werkzeug>=3.0.6" \
    "mlflow>=3.8.1" \
    "cbor2>=5.9.0" \
    "pyasn1>=0.6.3" \
    "xgrammar>=0.1.32"

# Clean build caches (saves 4-6 GB)
rm -rf /root/.cache/uv

# Strip CUDA object files left by compilation (saves ~250 MB)
find /opt/nemo_rl_venv -name '*.o' -delete 2>/dev/null || true

# Remove test suites from installed packages (saves ~130 MB)
find /opt/nemo_rl_venv -type d \( -name 'tests' -o -name 'test' \) \
    -path '*/site-packages/*' -exec rm -rf {} + 2>/dev/null || true

# Clean Ray aiohttp duplicate
find /opt/nemo_rl_venv -type d -name "aiohttp" \
    -path "*/ray/_private/*" -exec rm -rf {} + 2>/dev/null || true
EOF

ENV PATH="/opt/nemo_rl_venv/bin:$PATH"
ENV NEMO_RL_VENV_DIR=/opt/ray_venvs
WORKDIR /opt/nemo-rl


# ============================================================
# Stage 4: Full repo + Ray worker venvs + SBOM
# ============================================================
# Only 2 of 8 Ray worker venvs are prefetched (VllmGenerationWorker
# + DTensorPolicyWorkerV2). Saves ~60 GB vs full prefetch. Unused
# workers are created at first launch (~1 min each).

FROM hermetic AS release

ENV NEMO_RL_VENV_DIR=/opt/ray_venvs

COPY --from=nemo-rl-src . /opt/nemo-rl
RUN git rev-parse --is-shallow-repository | grep -q true && git fetch --unshallow || true

RUN <<"EOF" bash -exu
TORCH_CUDA_ARCH_LIST="9.0" uv run nemo_rl/utils/prefetch_venvs.py \
    VllmGenerationWorker DTensorPolicyWorkerV2 \
    --negative-filters sglang mcore

# Strip build artifacts from all venvs
find /opt/nemo_rl_venv -name '*.o' -delete
find /opt/ray_venvs -name '*.o' -delete 2>/dev/null || true
find /opt/nemo_rl_venv -type d \( -name 'tests' -o -name 'test' \) \
    -path '*/site-packages/*' -exec rm -rf {} + 2>/dev/null || true
find /opt/ray_venvs -type d \( -name 'tests' -o -name 'test' \) \
    -path '*/site-packages/*' -exec rm -rf {} + 2>/dev/null || true

rm -rf /root/.cache/uv
EOF

# CVE remediation: upgrade vulnerable packages across ALL venvs (main + Ray workers)
RUN <<"EOF" bash -exu
for venv in /opt/nemo_rl_venv /opt/ray_venvs/*/; do
  if [ -f "$venv/bin/python" ]; then
    uv pip install --python "$venv/bin/python" \
        "mlflow>=3.8.1" \
        "cryptography>=46.0.5" \
        "pyasn1>=0.6.3" \
        "wandb==0.26.0" \
        "ray==2.52.0" \
        2>/dev/null || true
  fi
done
rm -rf /root/.cache/uv
# Remove Go binary with CVE-2025-68121 (Nsight EFA metrics profiling plugin, not needed at runtime)
rm -f /usr/local/cuda-*/NsightSystems-cli-*/target-linux-x64/plugins/efa_metrics/nic_sampler
EOF

# Container fingerprint for cache invalidation
RUN python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint

# SBOM generation (produces /THIRD-PARTY-LICENSES and /SBOM.txt)
COPY generate_sbom.sh /tmp/generate_sbom.sh
RUN bash /tmp/generate_sbom.sh && rm /tmp/generate_sbom.sh


# ============================================================
# Stage 5: EFA networking stack
# ============================================================
# Installs the AWS EFA stack (libfabric 2.x + aws-ofi-nccl).
# Two modes:
#   EFA_INSTALL_MODE=full    — EFA installer (libfabric 2.4 + ofi-nccl 1.18)
#   EFA_INSTALL_MODE=minimal — Minimal install + manual ofi-nccl build
#
# P5en needs "full" mode for RDMA support via libfabric 2.x.

FROM release AS efa

ARG TARGETARCH
ARG EFA_INSTALL_MODE="full"
ARG EFA_INSTALLER_VERSION=1.47.0
ARG OFI_NCCL_VERSION=v1.18.0
ARG OFI_NCCL_PREFIX=/opt/aws-ofi-nccl

# --- Labels ---
LABEL maintainer="nemo-rl"
LABEL description="NeMo RL training image with EFA networking"
LABEL org.opencontainers.image.title="nemo-rl-workshop"
LABEL org.opencontainers.image.description="NeMo RL GRPO training with EFA, Ray, vLLM, DTensor"
LABEL org.opencontainers.image.vendor="nemo-rl"
LABEL org.opencontainers.image.architecture="linux/amd64"
LABEL nemo_rl.efa_mode="${EFA_INSTALL_MODE}"

# Build dependencies for EFA installer / aws-ofi-nccl compilation + GDRCopy
RUN apt-get update && apt-get install -y --no-install-recommends \
        autoconf automake libtool libhwloc-dev \
        environment-modules tcl \
        udev dmidecode \
        libevent-core-2.1-7t64 libevent-pthreads-2.1-7t64 libevent-dev \
    && rm -rf /var/lib/apt/lists/*

# GDRCopy — GPU Direct RDMA copy library (needed for EFA GDRDMA)
ARG GDRCOPY_VERSION=v2.5.1
RUN cd /tmp && \
    git clone --depth 1 --branch ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git && \
    cd gdrcopy && \
    make PREFIX=/usr lib lib_install && \
    ldconfig && \
    cd / && rm -rf /tmp/gdrcopy

# EFA installer — mode selection
# "full":    Non-minimal install gives libfabric 2.4 + ofi-nccl 1.18 from installer
# "minimal": Minimal install (libfabric only) + manual ofi-nccl build from source
RUN <<"EOF" bash -exu
if [ "${EFA_INSTALL_MODE}" = "full" ]; then
    # Full mode: purge old libfabric, install EFA with bundled ofi-nccl
    dpkg --purge --force-depends libfabric1-aws libfabric-aws-bin libfabric-aws-dev 2>/dev/null || true
    rm -rf /opt/amazon/efa
    cd /tmp
    curl -sL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz | tar xz
    cd aws-efa-installer
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify
    echo "EFA full install: $(/opt/amazon/efa/bin/fi_info --version 2>&1 | head -1)"
    cd / && rm -rf /tmp/aws-efa-installer
else
    # Minimal mode: libfabric only, build ofi-nccl from source
    cd /tmp
    curl -sL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz | tar xz
    cd aws-efa-installer
    ./efa_installer.sh -y --minimal --skip-kmod --skip-limit-conf --no-verify
    cd / && rm -rf /tmp/aws-efa-installer

    # Build aws-ofi-nccl from source
    git clone --depth 1 --branch ${OFI_NCCL_VERSION} \
        https://github.com/aws/aws-ofi-nccl.git /tmp/aws-ofi-nccl
    cd /tmp/aws-ofi-nccl
    ./autogen.sh
    ./configure \
        --prefix=${OFI_NCCL_PREFIX} \
        --with-libfabric=/opt/amazon/efa \
        --with-cuda=/usr/local/cuda \
        --with-nccl=/usr
    make -j"$(nproc)"
    make install
    rm -rf /tmp/aws-ofi-nccl
fi
EOF

# Clean stale aws-ofi-nccl from base image (we use the installer-provided NGC plugin)
RUN rm -rf /opt/amazon/aws-ofi-nccl 2>/dev/null || true

# Clean up build-only dependencies
RUN apt-get purge -y --auto-remove autoconf automake libtool && \
    rm -rf /var/lib/apt/lists/*

# Runtime library paths — adapt based on install mode
RUN <<"EOF" bash -exu
if [ "${EFA_INSTALL_MODE}" = "full" ]; then
    # Full mode: ofi-nccl is at /opt/amazon/ofi-nccl/lib/
    echo "/opt/amazon/ofi-nccl/lib" > /etc/ld.so.conf.d/ofi-nccl.conf
    echo "/opt/amazon/efa/lib" > /etc/ld.so.conf.d/efa.conf
else
    # Minimal mode: ofi-nccl is at custom prefix
    echo "${OFI_NCCL_PREFIX}/lib" > /etc/ld.so.conf.d/ofi-nccl.conf
    echo "/opt/amazon/efa/lib" > /etc/ld.so.conf.d/efa.conf
fi
ldconfig
EOF

ENV LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# EFA / NCCL defaults
ENV AWS_OFI_NCCL_VERSION=1.18.0
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1
ENV NCCL_NET_PLUGIN=ofi
ENV NCCL_TUNER_PLUGIN=ofi
ENV NCCL_SOCKET_IFNAME="^lo,docker,veth,eni"

# H200/NVSwitch NCCL workarounds (required for P5en, harmless on P5)
ENV NCCL_CUMEM_ENABLE=0
ENV NCCL_NVLS_ENABLE=0

# CVE remediation: upgrade system packages to latest security patches
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/nemo-rl
