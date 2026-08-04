# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# SGLang + DeepEP over NVSHMEM-libfabric (AWS EFA), for serving DeepSeek-R1
# with MoE expert parallelism on p5/p5en (8x H100/H200, sm_90) and p6-b300
# (8x B300, sm_103) nodes.
#
# Layers on top of the official SGLang image:
#   - EFA:      aws-efa-installer (libfabric + EFA provider)
#   - GDRCopy:  GPU memory registration for EFA VRAM transfer (NVSHMEM needs it)
#   - Mooncake: kvcache-ai @ main, built -DUSE_EFA=ON (KV transfer for PD-disagg)
#   - DeepEP:   deepseek-ai @ 567632d + the EFA patch, on NVSHMEM v3.7.0-0
#
# DeepEP and NVSHMEM are built by setup_deepep_efa.sh, which is a vendored copy
# of micro-benchmarks/expert-parallelism/deepep-benchmark/setup_deepep_efa.sh.
# See that benchmark's README for why the EFA patch is needed (DeepEP's
# internode kernels assume IBGDA, which EFA does not provide); this Dockerfile's
# README covers only the serving-engine integration.
#
# Build (no GPU needed; build context is this directory):
#   DOCKER_BUILDKIT=1 docker build -t sglang-deepep-efa:latest .

# The base must be a CUDA 13 (cu130) build. Beyond Blackwell codegen, CUDA <= 12.9
# CUPTI returns CUPTI_ERROR_INVALID_DEVICE on B200/B300, which breaks the
# internode / low-latency tuning phase (it profiles kernels) -- correctness checks
# pass first, so the failure looks unrelated. Do not downgrade for Blackwell.
ARG SGLANG_BASE=lmsysorg/sglang:v0.5.13.post1-cu130
FROM ${SGLANG_BASE}

USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        git cmake build-essential wget curl \
        libgflags-dev autoconf automake libtool ninja-build
# NOTE: do NOT purge /var/lib/apt/lists here — the EFA installer below runs its
# own `apt-get install` (pciutils, tcl, environment-modules, ...) and needs the
# package index. It is purged at the end of the EFA step instead.

# ---- GDRCopy (GPU memreg for EFA, required by NVSHMEM + Mooncake) ----
ARG GDRCOPY_VERSION=2.5.2
RUN cd /tmp && \
    wget -q https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz && \
    tar xf v${GDRCOPY_VERSION}.tar.gz && cd gdrcopy-${GDRCOPY_VERSION} && \
    make -j$(nproc) lib lib_install CUDA=/usr/local/cuda PREFIX=/usr/local && \
    rm -rf /tmp/gdrcopy* /tmp/v${GDRCOPY_VERSION}.tar.gz

# ---- AWS EFA installer (libfabric + EFA provider) ----
# Bundled rdma-core + --skip-rdma-core so the build does not depend on whether
# the build host has EFA devices. DEBS dir is matched to the base image OS.
# Pinned, not `latest`: libfabric, the EFA provider and the aws-ofi-nccl plugin
# are the stack this sample measures, so an unpinned rebuild can move the numbers
# with no change to this repo. 1.49.0 ships libfabric 2.4.0amzn5.0 and
# libnccl-ofi-ngc-v3 1.20.0; the image records the full set in
# /opt/amazon/efa_installed_packages.
ARG EFA_INSTALLER_VERSION=1.49.0
RUN cd /tmp && \
    curl -fsSL -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    tar xzf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    . /etc/os-release && \
    case "$VERSION_ID" in \
      24.04) DEBS_DIR=UBUNTU2404 ;; \
      22.04) DEBS_DIR=UBUNTU2204 ;; \
      *)     DEBS_DIR=UBUNTU2404 ;; \
    esac && \
    echo "Using EFA DEBS dir: $DEBS_DIR (Ubuntu $VERSION_ID)" && \
    (apt-get install -y ./DEBS/${DEBS_DIR}/x86_64/rdma-core/*.deb || true) && \
    ./efa_installer.sh -y --skip-kmod -g --no-verify --skip-rdma-core && \
    rm -rf /tmp/aws-efa-installer* /var/lib/apt/lists/*

# ---- Mooncake transfer engine (EFA + CUDA) ----
# Any mooncake already in the base image is a pip wheel built WITHOUT EFA
# support, so it is replaced with a -DUSE_EFA=ON source build.
#
# MOONCAKE_COMMIT must be >= a7413723 ("[TE] fix(efa): bind a CUDA context before
# registering GPU memory", kvcache-ai/Mooncake#3177, merged 2026-07-30) or the
# 2P2D KV path does not work at all. Before that commit, fi_mr_regattr() on
# FI_HMEM_CUDA memory fails with a bare "Operation not supported": libfabric's
# cuda_get_dmabuf_fd() calls cuMemGetHandleForAddressRange() without binding a
# context, so a thread that has only used the CUDA runtime API gets
# CUDA_ERROR_INVALID_CONTEXT, and the EFA provider -- which has no dmabuf
# fallback for FI_HMEM_CUDA -- reports it as ENOTSUP. registerLocalMemory() then
# rolls the chunk back, leaving the buffer registered on no NIC.
#
# The failure is invisible until it matters: startup finishes, SGLang's PD warmup
# passes (its 4-token prompt does not exercise the path), and the first real
# request dies with "remote mooncake session ... is not alive" -- blaming the peer,
# which is alive. Do not go looking for env levers; check this pin first.
ARG MOONCAKE_COMMIT=a7413723
RUN pip uninstall -y mooncake-transfer-engine mooncake 2>/dev/null || true
RUN cd /tmp && \
    git clone https://github.com/kvcache-ai/Mooncake.git && \
    cd Mooncake && git checkout ${MOONCAKE_COMMIT} && git log --oneline -1 && \
    bash dependencies.sh -y && \
    apt-get install -y libgflags-dev && \
    mkdir build && cd build && \
    PYBIN=$(which python3) && \
    PYINC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") && \
    cmake .. -DUSE_EFA=ON -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DPython3_EXECUTABLE="$PYBIN" \
      -DPython3_INCLUDE_DIR="$PYINC" \
      -DPython_EXECUTABLE="$PYBIN" && \
    make -j$(nproc) && \
    cp mooncake-integration/engine.cpython-*.so ../mooncake-wheel/mooncake/ && \
    { cp mooncake-integration/store.cpython-*.so ../mooncake-wheel/mooncake/ 2>/dev/null || true; } && \
    cp mooncake-common/libasio.so ../mooncake-wheel/mooncake/ && \
    pip install ../mooncake-wheel --no-build-isolation && \
    MCDIR=$(python3 -c "import mooncake, os; print(os.path.dirname(mooncake.__file__))") && \
    cp -v mooncake-integration/engine.cpython-*.so "$MCDIR/" && \
    { cp -v mooncake-integration/store.cpython-*.so "$MCDIR/" 2>/dev/null || true; } && \
    cp -v mooncake-common/libasio.so "$MCDIR/" && \
    rm -rf /tmp/Mooncake

# ---- DeepEP + NVSHMEM (EFA / libfabric transport) ----
# Pins enforced by setup_deepep_efa.sh:
#   DeepEP  == 567632d   (pre-EPv2; the script hard-checks the tree)
#   NVSHMEM == v3.7.0-0  (upstream NVIDIA release tag)
ARG DEEPEP_REPO=https://github.com/deepseek-ai/DeepEP.git
ARG DEEPEP_COMMIT=567632d
ARG NVSHMEM_REPO=https://github.com/NVIDIA/nvshmem.git
ARG NVSHMEM_TAG=v3.7.0-0
# Hopper + Blackwell, matching the canonical benchmark image so one image runs on
# p5/p5en (sm_90) and p6-b300 (sm_103; B200 is sm_100).
#
# Two consequences of the multi-arch default, both intentional:
#   - setup_deepep_efa.sh enables DeepEP's aggressive PTX instructions ONLY when
#     the list is exactly "9.0" (they are Hopper-specific), so a multi-arch build
#     compiles Hopper without them. The H100/H200 numbers in benchmarks/RESULTS.md
#     were measured from a `9.0` build -- pass that build-arg to reproduce them
#     bit-for-bit.
#   - three architectures roughly triples the DeepEP/NVSHMEM device-code compile.
# Single-arch builds: 9.0 (Hopper), 10.0 (B200), 10.3 (B300).
ARG TORCH_CUDA_ARCH_LIST="9.0;10.0;10.3"

ENV DEEPEP_SRC=/opt/DeepEP
ENV NVSHMEM_SRC=/opt/nvshmem
ENV NVSHMEM_PREFIX=/opt/nvshmem/install
ENV DEEPEP_VENV=/opt/deepep-venv

# CUDA 13 relocated the libcu++ headers (cuda/std/array, included by NVSHMEM's
# device headers) into include/cccl. nvcc finds them automatically, but the host
# g++ compile of deep_ep.cpp does not — it fails with
# "fatal error: cuda/std/array: No such file or directory". CPATH is honored by
# both g++ and nvcc, so add the cccl dir there for the DeepEP build.
# Set rather than appended: the base image leaves CPATH empty, and appending an
# empty value would leave a trailing colon, which g++ reads as "also search .".
ENV CPATH=/usr/local/cuda/include/cccl

COPY setup_deepep_efa.sh /opt/setup_deepep_efa.sh

RUN git clone ${DEEPEP_REPO} ${DEEPEP_SRC} && \
    cd ${DEEPEP_SRC} && git checkout ${DEEPEP_COMMIT} && \
    git clone ${NVSHMEM_REPO} ${NVSHMEM_SRC} && \
    cd ${NVSHMEM_SRC} && git checkout ${NVSHMEM_TAG}

# The SGLang base image may ship a stock `deep_ep` built for the standard
# NVSHMEM IBGDA transport (InfiniBand, which EFA does not provide). It lives in
# the system dist-packages, which sits *ahead* of the venv path added by the
# .pth below on sys.path, so leaving it would silently shadow the EFA build —
# you would believe you are running on EFA while actually running IBGDA.
RUN pip uninstall -y deep_ep deep-ep 2>/dev/null || true

# DeepEP is built into a --system-site-packages venv so it compiles against the
# base image's PyTorch without pip writing into the base image's dist-packages,
# then exposed to the image's default python3 (which the sglang launcher uses)
# via a .pth. The build asserts deep_ep resolves into the venv, so a leftover
# system copy fails the build instead of silently shadowing the EFA one.
RUN python3 -m venv --system-site-packages ${DEEPEP_VENV} && \
    chmod +x /opt/setup_deepep_efa.sh && \
    /opt/setup_deepep_efa.sh \
        --deepep-src      ${DEEPEP_SRC} \
        --nvshmem-src     ${NVSHMEM_SRC} \
        --nvshmem-prefix  ${NVSHMEM_PREFIX} \
        --python          ${DEEPEP_VENV}/bin/python \
        --pip             ${DEEPEP_VENV}/bin/pip \
        --cuda-home       /usr/local/cuda \
        --libfabric-home  /opt/amazon/efa \
        --gdrcopy-home    /usr/local && \
    VENV_SP=$(${DEEPEP_VENV}/bin/python -c "import sysconfig; print(sysconfig.get_path('purelib'))") && \
    SYS_SP=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))") && \
    echo "$VENV_SP" > "$SYS_SP/deepep_efa.pth" && \
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    TORCH_LIB=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')") && \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${TORCH_LIB}:${NVSHMEM_PREFIX}/lib:/opt/amazon/efa/lib:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    python3 -c "import deep_ep, os; p=os.path.dirname(deep_ep.__file__); print('deep_ep ->', p); assert p.startswith('${DEEPEP_VENV}'), 'WRONG deep_ep on path: '+p"

# ---- Resolve the NVSHMEM version collision ----
# The base image ships pip `nvidia-nvshmem-cu13` (v3.4.5) as a dependency of
# torch's libtorch_nvshmem.so. Its libnvshmem_host.so.3 is loaded (via torch)
# AHEAD of ours, but DeepEP's device cubin is compiled against NVSHMEM v3.7.0 ->
# "NVSHMEM device library version does not match host library version" at
# nvshmem init, aborting every internode / low-latency run.
# Fix: overwrite the pip host lib with our v3.7.0 build, so the single
# libnvshmem_host.so.3 soname resolves to v3.7.0 for BOTH torch_nvshmem and
# deep_ep. Replacing rather than uninstalling keeps torch_nvshmem loadable.
# nvidia.nvshmem is a namespace package, so it has no __file__ — locate it via
# find_spec's search locations instead.
RUN PIP_NVSHMEM=$(python3 -c "import importlib.util as u; s = u.find_spec('nvidia.nvshmem'); print(list(s.submodule_search_locations)[0] + '/lib' if s else '')" 2>/dev/null || true) && \
    echo "pip nvshmem lib dir: '${PIP_NVSHMEM:-<none>}'" && \
    if [ -n "$PIP_NVSHMEM" ] && [ -e "$PIP_NVSHMEM/libnvshmem_host.so.3" ]; then \
        cp -f ${NVSHMEM_PREFIX}/lib/libnvshmem_host.so.3.7.0 "$PIP_NVSHMEM/libnvshmem_host.so.3" && \
        echo "Replaced pip nvshmem host lib with v3.7.0:" && \
        strings "$PIP_NVSHMEM/libnvshmem_host.so.3" | grep -m1 "NVSHMEM v3" ; \
    else echo "no pip nvidia-nvshmem present, skipping"; fi

# ---- EFA + NVSHMEM runtime env ----
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1
ENV NVSHMEM_DIR=${NVSHMEM_PREFIX}
ENV NVSHMEM_HOME=${NVSHMEM_PREFIX}
# NVSHMEM is built with ONLY the libfabric remote transport (IBRC/IBGDA off), so
# the default REMOTE_TRANSPORT=ibrc would find no transport ("Peer GPU not
# accessible / building transport map failed"). Force libfabric+efa.
ENV NVSHMEM_REMOTE_TRANSPORT=libfabric
ENV NVSHMEM_LIBFABRIC_PROVIDER=efa
# NVSHMEM 3.7 + EFA: give each GPU exclusive use of the NIC on its own PCIe
# switch for best RDMA performance (p5/p5en pair multiple EFA NICs across PCIe
# switches). Recommended by the NVSHMEM/EFA integration.
ENV NVSHMEM_NETDEVS_POLICY=EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE

# ---- NCCL aws-ofi-nccl plugin name ----
# On NGC-based images the EFA installer lays down libnccl-ofi-ngc-v3, which ships
# ONLY libnccl-net-ofi.so -- NOT the default name libnccl-net.so that NCCL
# auto-loads. Without this, NCCL logs "NET/Plugin: Could not find: libnccl-net.so"
# and falls back to TCP sockets (~14 GB/s vs ~400 GB/s over EFA), which shows up as
# 3-5x slower prefill TTFT rather than as an error.
# Use the SHORT name "ofi": NCCL templates it into libnccl-net-<value>.so and
# resolves that through ldconfig, where the EFA installer registered
# /opt/amazon/ofi-nccl/lib. Do NOT use an absolute path -- NCCL 2.27.x applies the
# same templating to it, yielding the bogus
# "libnccl-net-/opt/.../libnccl-net-ofi.so.so" -> not found -> silent TCP fallback.
# (2.28+ happens to accept an absolute path, so this matters only if the base image
# moves back to 2.27, but "ofi" is correct on both.)
# Verify: NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET should log
# "NET/OFI Selected provider is efa" -- not "NET/Socket".
ENV NCCL_NET_PLUGIN=ofi
# NOTE on CUDA VMM: it is REGIME-SPECIFIC and therefore NOT set here.
#   - normal dispatch/combine (test_intranode / test_internode): set
#     NVSHMEM_DISABLE_CUDA_VMM=1 at runtime, else NVSHMEM topo / transport-map
#     init fails inside the container.
#   - low_latency (test_low_latency): leave CUDA VMM ENABLED, else the RDMA
#     buffer cudaMemset fails with "invalid argument" (deep_ep.cpp:371).
# recipe/run-kernel-test.sh sets this per test; see README.
ENV PATH="/opt/amazon/efa/bin:${PATH}"
# TORCH_LIB is the base image's torch/lib. It is resolved at build time and
# baked in below, because ENV cannot run a command; the sanity check in
# recipe/verify-image.sh asserts the resulting path exists.
ENV LD_LIBRARY_PATH="${NVSHMEM_PREFIX}/lib:/opt/amazon/efa/lib:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN TORCH_LIB=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')") && \
    printf '%s\n' "$TORCH_LIB" > /etc/ld.so.conf.d/zz-torch.conf && \
    printf '%s\n' "${NVSHMEM_PREFIX}/lib" > /etc/ld.so.conf.d/zz-nvshmem.conf && \
    ldconfig && \
    echo "registered torch lib dir with ldconfig: $TORCH_LIB"

WORKDIR /workspace
