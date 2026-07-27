# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# vLLM + DeepEP over NVSHMEM-libfabric (AWS EFA), for serving DeepSeek-R1 with
# MoE expert parallelism on p5/p5en nodes (8x H100/H200, sm_90).
#
# Layers on top of the official vLLM image:
#   - CUDA dev headers: the vLLM base ships a SLIM toolkit; NVSHMEM/DeepEP need
#                       nvml.h, cusparse.h and nvtx3/ which are absent
#   - EFA:      aws-efa-installer (libfabric + EFA provider)
#   - GDRCopy:  GPU memory registration for EFA VRAM transfer (NVSHMEM needs it)
#   - Mooncake: kvcache-ai @ main, built -DUSE_EFA=ON (optional KV connector)
#   - DeepEP:   deepseek-ai @ 567632d + the EFA patch, on NVSHMEM v3.7.0-0
#
# NIXL is NOT built here: the base image already ships nixl_cu13 with
# libplugin_LIBFABRIC.so, which is what carries KV over EFA for PD-disaggregation.
#
# DeepEP and NVSHMEM are built by setup_deepep_efa.sh, which is a vendored copy
# of micro-benchmarks/expert-parallelism/deepep-benchmark/setup_deepep_efa.sh.
# See that benchmark's README for why the EFA patch is needed (DeepEP's
# internode kernels assume IBGDA, which EFA does not provide); this Dockerfile's
# README covers only the serving-engine integration.
#
# Build (no GPU needed; build context is this directory):
#   DOCKER_BUILDKIT=1 docker build -t vllm-deepep-efa:latest .

ARG VLLM_BASE=vllm/vllm-openai:v0.23.0
FROM ${VLLM_BASE}

USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        git cmake build-essential wget curl \
        libgflags-dev autoconf automake libtool ninja-build && \
    (apt-get install -y libhwloc15 || apt-get install -y hwloc-nox || true)
# NOTE: do NOT purge /var/lib/apt/lists here — both the header gap-fill and the
# EFA installer below run their own `apt-get install` and need the package index.
# It is purged at the end of the EFA step instead.

# ---- Fill the gaps in the base image's slim CUDA toolkit ----
# NVSHMEM and DeepEP include nvml.h, cusparse.h and nvtx3/, none of which ship in
# the vLLM base's toolkit. Two sources: nvml.h from the CUDA apt repo already
# configured in the base, everything else from the nvidia-cu13 pip package.
#
# CRITICAL: symlink ONLY the headers the toolkit is MISSING. Do not put the whole
# pip include dir on CPATH/-I and do not overwrite existing toolkit headers — the
# pip dir carries a parallel copy of the CUDA runtime headers (cuda_runtime.h,
# crt/host_runtime.h) which, if they shadow the toolkit's, break nvcc's generated
# stubs with "'__cudaLaunch' was not declared".
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvml-dev-13-0 || true
RUN PIP_INC=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include && \
    DST=/usr/local/cuda/include && \
    if [ -d "$PIP_INC" ]; then \
      for f in "$PIP_INC"/*.h "$PIP_INC"/*.hpp "$PIP_INC"/nvtx3; do \
        b=$(basename "$f"); \
        [ -e "$DST/$b" ] || ln -s "$f" "$DST/$b"; \
      done; \
    fi && \
    echo "#include <cusparse.h>" | gcc -I"$DST" -E -x c - >/dev/null && \
    echo "#include <nvml.h>"     | gcc -I"$DST" -E -x c - >/dev/null && \
    echo "gap-fill headers OK (cusparse.h + nvml.h resolvable)"

# ---- GDRCopy (GPU memreg for EFA, required by NVSHMEM + Mooncake) ----
ARG GDRCOPY_VERSION=2.5.2
RUN cd /tmp && \
    wget -q https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz && \
    tar xf v${GDRCOPY_VERSION}.tar.gz && cd gdrcopy-${GDRCOPY_VERSION} && \
    make -j$(nproc) lib lib_install CUDA=/usr/local/cuda PREFIX=/usr/local && \
    rm -rf /tmp/gdrcopy* /tmp/v${GDRCOPY_VERSION}.tar.gz

# ---- AWS EFA installer (libfabric + EFA provider) ----
# Bundled rdma-core + --skip-rdma-core so the build does not depend on whether
# the build host has EFA devices. The DEBS dir MUST match the base OS: the vLLM
# base is Ubuntu 22.04 (jammy) while the sglang base is 24.04, and the 24.04
# rdma-core deb needs glibc >= 2.38 where jammy has 2.35.
ARG EFA_INSTALLER_VERSION=latest
RUN cd /tmp && \
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    tar xzf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    . /etc/os-release && \
    case "$VERSION_ID" in \
      24.04) DEBS_DIR=UBUNTU2404 ;; \
      22.04) DEBS_DIR=UBUNTU2204 ;; \
      *)     DEBS_DIR=UBUNTU2204 ;; \
    esac && \
    echo "Using EFA DEBS dir: $DEBS_DIR (Ubuntu $VERSION_ID)" && \
    (apt-get install -y ./DEBS/${DEBS_DIR}/x86_64/rdma-core/*.deb || true) && \
    ./efa_installer.sh -y --skip-kmod -g --no-verify --skip-rdma-core && \
    rm -rf /tmp/aws-efa-installer* /var/lib/apt/lists/*

# Expose libcuda for build-time linking. Unlike the sglang base, this image keeps
# libcuda.so ONLY under lib64/stubs, which is not a default linker search path —
# DeepEP's final link passes -lcuda with -L/usr/local/cuda/lib64 and fails with
# "/usr/bin/ld: cannot find -lcuda". Two symlinks are needed:
#   lib64/libcuda.so      -> so `-lcuda` resolves at link time
#   stubs/libcuda.so.1    -> so the recorded soname resolves when the freshly
#                            built deep_ep is imported in this GPU-less build
#                            (stubs is put on LD_LIBRARY_PATH for that check only,
#                            never in the image's runtime LD_LIBRARY_PATH, where
#                            it would shadow the real driver).
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so && \
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    ldconfig -p >/dev/null && echo "libcuda link stubs in place"

# ---- Mooncake transfer engine (EFA + CUDA) ----
# Not on the PD path used by this sample's recipes (that is NixlConnector over
# LIBFABRIC), but built here so the image can also drive vLLM's MooncakeConnector
# and to keep it interchangeable with the sglang sample's image. Any mooncake
# already in the base is a pip wheel built WITHOUT EFA support, so it is replaced
# with a -DUSE_EFA=ON source build. Pin Python3_INCLUDE_DIR to the active 3.12
# headers so the .so is not tagged with a stale ABI.
RUN pip uninstall -y mooncake-transfer-engine mooncake 2>/dev/null || true
RUN cd /tmp && \
    git clone --depth 1 https://github.com/kvcache-ai/Mooncake.git && \
    cd Mooncake && git log --oneline -1 && \
    bash dependencies.sh -y && \
    apt-get install -y libgflags-dev && \
    mkdir build && cd build && \
    PYBIN=$(which python3) && \
    PYINC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") && \
    echo "BUILD-PYTHON: $PYBIN $($PYBIN --version) inc=$PYINC" && \
    cmake .. -DUSE_EFA=ON -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DPython3_EXECUTABLE="$PYBIN" \
      -DPython3_INCLUDE_DIR="$PYINC" \
      -DPython_EXECUTABLE="$PYBIN" && \
    make -j$(nproc) && \
    cp mooncake-integration/engine.cpython-*.so ../mooncake-wheel/mooncake/ && \
    cp mooncake-integration/store.cpython-*.so ../mooncake-wheel/mooncake/ 2>/dev/null || true && \
    cp mooncake-common/libasio.so ../mooncake-wheel/mooncake/ && \
    pip install ../mooncake-wheel --no-build-isolation && \
    MCDIR=$(python3 -c "import mooncake, os; print(os.path.dirname(mooncake.__file__))") && \
    cp -v mooncake-integration/engine.cpython-*.so "$MCDIR/" && \
    cp -v mooncake-integration/store.cpython-*.so "$MCDIR/" 2>/dev/null || true && \
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
# Hopper only: this image is validated on p5 (H100) and p5en (H200). The
# benchmark image supports Blackwell too, but that is not exercised here.
ARG TORCH_CUDA_ARCH_LIST=9.0

ENV DEEPEP_SRC=/opt/DeepEP
ENV NVSHMEM_SRC=/opt/nvshmem
ENV NVSHMEM_PREFIX=/opt/nvshmem/install
ENV DEEPEP_VENV=/opt/deepep-venv

# CUDA 13 relocated the libcu++ headers (cuda/std/array, included by NVSHMEM's
# device headers) into include/cccl. nvcc finds them automatically, but the host
# g++ compile of deep_ep.cpp does not — it fails with
# "fatal error: cuda/std/array: No such file or directory". CPATH is honored by
# both g++ and nvcc, so add ONLY the cccl dir here (nvtx3 is handled by the
# symlink above; adding the whole pip include dir would shadow runtime headers).
# Set rather than appended: the base image leaves CPATH empty, and appending an
# empty value would leave a trailing colon, which g++ reads as "also search .".
ENV CPATH=/usr/local/cuda/include/cccl

COPY setup_deepep_efa.sh /opt/setup_deepep_efa.sh

RUN git clone ${DEEPEP_REPO} ${DEEPEP_SRC} && \
    cd ${DEEPEP_SRC} && git checkout ${DEEPEP_COMMIT} && \
    git clone ${NVSHMEM_REPO} ${NVSHMEM_SRC} && \
    cd ${NVSHMEM_SRC} && git checkout ${NVSHMEM_TAG}

# The vLLM base image ships its own `deep_ep` in the system dist-packages, built
# for the standard NVSHMEM IBGDA transport (InfiniBand, which EFA does not
# provide). That directory sits *ahead* of the venv path added by the .pth below
# on sys.path, so leaving it would silently shadow the EFA build — you would
# believe you are running on EFA while actually running IBGDA.
RUN pip uninstall -y deep_ep deep-ep 2>/dev/null || true

# DeepEP is built into a --system-site-packages venv, which compiles against the
# base image's PyTorch without pip writing into the base image's dist-packages,
# then is exposed to the image's default python3 (which vllm/ray use) via a .pth.
# Unlike the sglang base, this base is NOT PEP 668 externally-managed, so a venv
# is not strictly required here — it is used because it puts the EFA build at a
# *distinguishable path*: the assert below then proves the import resolves to our
# build and not to a leftover or later-reinstalled stock copy in dist-packages.
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
        strings "$PIP_NVSHMEM/libnvshmem_host.so.3" | grep -o "NVSHMEM v3\.[0-9]*\.[0-9]*" | sort -u ; \
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
# switches). Worth +8-12% on p5en (16 NICs); a no-op on p5 (32 NICs).
ENV NVSHMEM_NETDEVS_POLICY=EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE
# NOTE on CUDA VMM: it is REGIME-SPECIFIC and therefore NOT set here.
#   - normal dispatch/combine (test_intranode / test_internode): set
#     NVSHMEM_DISABLE_CUDA_VMM=1 at runtime, else NVSHMEM topo / transport-map
#     init fails inside the container.
#   - low_latency (test_low_latency): leave CUDA VMM ENABLED, else the RDMA
#     buffer cudaMemset fails with "invalid argument" (deep_ep.cpp:371).
# recipe/run-kernel-test.sh sets this per test; see README.
ENV PATH="/opt/amazon/efa/bin:${PATH}"
# The nixl_cu13 lib dirs carry libplugin_LIBFABRIC.so and its deps; NixlConnector
# only finds them if both are on the loader path.
ENV NIXL_LIBS=/usr/local/lib/python3.12/dist-packages/nixl_cu13.libs
ENV NIXL_PLUGIN_DIR=/usr/local/lib/python3.12/dist-packages/nixl_cu13.libs/nixl
ENV LD_LIBRARY_PATH="${NVSHMEM_PREFIX}/lib:/opt/amazon/efa/lib:/usr/local/lib:/usr/local/cuda/lib64:${NIXL_LIBS}:${NIXL_PLUGIN_DIR}:${LD_LIBRARY_PATH}"
# torch/lib is resolved at build time and registered with ldconfig, because ENV
# cannot run a command; recipe/verify-image.sh asserts the path exists.
RUN TORCH_LIB=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')") && \
    printf '%s\n' "$TORCH_LIB" > /etc/ld.so.conf.d/zz-torch.conf && \
    printf '%s\n' "${NVSHMEM_PREFIX}/lib" > /etc/ld.so.conf.d/zz-nvshmem.conf && \
    ldconfig && \
    echo "registered torch lib dir with ldconfig: $TORCH_LIB"

# Ray is the only supported multi-node executor for TP=16 in this sample (see
# README): vllm's own --nnodes/--node-rank fails with "collective_rpc should not
# be called on follower node".
#
# vllm-router is a SEPARATE PyPI package, not part of the vLLM image. It is baked
# in here rather than pip-installed at run time so the PD front end is
# reproducible. The bundled examples/disaggregated/.../disagg_proxy_demo.py is a
# toy and collapses at concurrency >= 16, so it is not a substitute.
ARG VLLM_ROUTER_VERSION=0.1.15
RUN pip install --no-cache-dir ray "vllm-router==${VLLM_ROUTER_VERSION}" && \
    which vllm-router && vllm-router --version 2>/dev/null || true

WORKDIR /workspace
ENTRYPOINT []
