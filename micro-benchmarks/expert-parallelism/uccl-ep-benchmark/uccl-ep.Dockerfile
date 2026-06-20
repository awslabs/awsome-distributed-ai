# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# UCCL expert-parallelism micro-benchmark image, B300-compatible.
#
# The UCCL-EP kernels are built with `python3 setup.py install` and an explicit
# multi-arch PTX list (Hopper sm_90 + Blackwell sm_100/sm_103), following the
# validated build in the dsv3-uccl-nixl recipe
# (gitlab.aws.dev/mlkeita/2026-06-pavel-blog-nccl-nixl). This replaces the
# upstream `ep/install_deps.sh && make` path, which (a) compiles a single
# auto-detected arch and (b) fails its torch verify step in a GPU-less Docker
# build.
ARG CUDA_VERSION=13.0.3
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

ARG GDRCOPY_VERSION=v2.5.2
ARG EFA_INSTALLER_VERSION=1.48.0
ARG NCCL_VERSION=v2.30.4-1
ARG NCCL_TESTS_VERSION=v2.18.3
# Pin UCCL to a known-good commit for reproducible, B300-validated builds (same
# commit as the dsv3-uccl-nixl recipe).
ARG UCCL_COMMIT=0dc87eb3b40c372a16b70ef320f37daaa5299ca7

# --- system packages ---------------------------------------------------------
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get remove -y --allow-change-held-packages \
    ibverbs-utils libibverbs-dev libibverbs1 libmlx5-1 libnccl2 libnccl-dev || true

RUN rm -rf /opt/hpcx /usr/local/mpi \
    && rm -f /etc/ld.so.conf.d/hpcx.conf \
    && ldconfig

ENV OPAL_PREFIX=

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils autoconf automake build-essential check cmake curl debhelper \
    devscripts git gcc gdb kmod libsubunit-dev libtool openssh-client \
    openssh-server pkg-config vim python3-dev python3-pip python3-venv

RUN apt-get purge -y cuda-compat-* || true

RUN mkdir -p /var/run/sshd
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH

RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED /usr/lib/python3/EXTERNALLY-MANAGED
RUN pip3 install --break-system-packages awscli pynvml

# --- GDRCopy -----------------------------------------------------------------
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix=/opt/gdrcopy install

ENV LD_LIBRARY_PATH=/opt/gdrcopy/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/gdrcopy/lib:$LIBRARY_PATH
ENV CPATH=/opt/gdrcopy/include:$CPATH
ENV PATH=/opt/gdrcopy/bin:$PATH

# --- EFA installer (bundles the AWS OFI NCCL plugin) -------------------------
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify --disable-ngc \
    && rm -rf $HOME/aws-efa-installer

# --- NCCL (CUDA arches 8.0/8.6/8.9/9.0/10.0/10.3) ----------------------------
RUN git clone -b ${NCCL_VERSION} https://github.com/NVIDIA/nccl.git /opt/nccl \
    && cd /opt/nccl \
    && make -j $(nproc) src.build CUDA_HOME=/usr/local/cuda \
       NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_103,code=sm_103"

# --- NCCL tests (handy for fi_info / health checks) --------------------------
RUN git clone -b ${NCCL_TESTS_VERSION} https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && cd /opt/nccl-tests \
    && make -j $(nproc) MPI=1 MPI_HOME=/opt/amazon/openmpi/ \
       CUDA_HOME=/usr/local/cuda NCCL_HOME=/opt/nccl/build \
       NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_103,code=sm_103"

RUN rm -rf /var/lib/apt/lists/*

ENV OMPI_MCA_pml=^ucx \
    OMPI_MCA_btl=tcp,self \
    OMPI_MCA_btl_tcp_if_exclude=lo,docker0,veth_def_agent \
    OPAL_PREFIX=/opt/amazon/openmpi \
    NCCL_SOCKET_IFNAME=^docker,lo,veth \
    PMIX_MCA_gds=hash \
    LD_PRELOAD=/opt/nccl/build/lib/libnccl.so

# --- Python tooling (uv) + UCCL build deps -----------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV UV_HTTP_TIMEOUT=500
ENV UV_LINK_MODE=copy

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ccache libibverbs-dev libnl-3-dev libnl-route-3-dev libnuma-dev ninja-build \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (CUDA 13 wheels) — required to build and run the UCCL-EP kernels.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system torch numpy \
        --extra-index-url https://download.pytorch.org/whl/cu130

# --- UCCL-EP -----------------------------------------------------------------
RUN git clone https://github.com/uccl-project/uccl.git /opt/uccl \
    && cd /opt/uccl \
    && git checkout ${UCCL_COMMIT} \
    && git submodule update --init --recursive

RUN uv pip install --system nanobind
WORKDIR /opt/uccl
RUN uv pip install --system .

# UCCL-EP uses SM90+ features (TMA, mbarrier); the PTX-augmented arch list runs
# on Hopper (sm_90) and Blackwell (sm_100/sm_103) without a per-arch rebuild.
WORKDIR /opt/uccl/ep
ENV PER_EXPERT_BATCHING=1
RUN TORCH_CUDA_ARCH_LIST="9.0a+PTX;10.0a+PTX;10.3a+PTX" python3 setup.py install

# --- smoke test --------------------------------------------------------------
RUN python3 -c "import torch; import uccl.ep; print('uccl.ep OK')"

WORKDIR /opt/uccl/ep
