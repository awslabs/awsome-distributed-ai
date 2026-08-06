# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# vLLM serving image with DeepEP expert-parallel all-to-all over EFA.
#
# Layers EFA + GDRCopy + NVSHMEM 3.7.0 + DeepEP v1 (EFA-patched) onto the stock
# vLLM release image, following the build in
# micro-benchmarks/expert-parallelism/deepep-benchmark/.
#
# Why layer instead of building vLLM from source: vllm/vllm-openai:v0.21.0
# already ships the full CUDA 13.0 devel toolkit (nvcc 13.0.88), torch
# 2.11.0+cu130, ninja and CUDA's cccl headers -- every prerequisite
# setup_deepep_efa.sh checks for except EFA itself. Rebuilding vLLM from source
# would add hours of compile time and risk drift from the published release that
# the existing benchmarks in this repo were measured against.
#
# Build (context must be the deepep-benchmark directory so that the single
# source of truth for setup_deepep_efa.sh is used, not a copy):
#
#   cd micro-benchmarks/expert-parallelism/deepep-benchmark
#   DOCKER_BUILDKIT=1 docker build --progress=plain \
#       -f ../../../3.test_cases/pytorch/vllm/vllm-deepep-efa/vllm-deepep-efa.Dockerfile \
#       -t vllm-deepep:efa1.48.0 .
#
# or use ./build.sh, which does the above and the enroot import.

ARG VLLM_VERSION=v0.21.0
FROM vllm/vllm-openai:${VLLM_VERSION}

ARG GDRCOPY_VERSION=v2.5.2
ARG EFA_INSTALLER_VERSION=1.48.0
ARG NVSHMEM_VERSION=3.7.0
ARG DEEPEP_COMMIT=567632d

# p5en/p5 are H200/H100 (sm_90). Hopper-only keeps the image small and, because
# setup_deepep_efa.sh only enables DeepEP's aggressive PTX instructions when the
# arch list is exactly "9.0", it is also the faster build. For Blackwell use
# --build-arg TORCH_CUDA_ARCH_LIST="10.0" (or "9.0;10.0" for a portable image).
ARG TORCH_CUDA_ARCH_LIST="9.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

# Used ONLY to import CUDA-linked extensions during build-time smoke tests, where
# the nvidia container runtime has not injected the host driver. Never added to
# the image's LD_LIBRARY_PATH -- at runtime the injected driver must win.
# Note the stubs dir cannot serve this purpose: its file is named libcuda.so
# while its SONAME is libcuda.so.1, so the loader does not resolve it (verified).
ARG CUDA_COMPAT_DIR=/usr/local/cuda-13.0/compat

# --- build prerequisites -----------------------------------------------------
# The base image has ninja and cccl but no git (needed to clone DeepEP) and no
# cmake. Conflicting distro RDMA packages are removed first: the EFA installer
# ships its own libibverbs/librdmacm and rdma-core here would shadow them.
# The apt index is deliberately NOT pruned here: the EFA installer runs its own
# apt-get install for third-party deps (tcl, among others) and fails with
# "Unable to locate package tcl" if the lists have been removed. It is cleaned up
# after the EFA layer instead.
RUN apt-get update -y \
    && apt-get remove -y --allow-change-held-packages \
        ibverbs-utils libibverbs-dev libibverbs1 libmlx5-1 || true \
    && apt-get install -y --no-install-recommends \
        autoconf automake build-essential cmake curl git libtool pkg-config \
        libnl-3-dev libnl-route-3-dev libnuma-dev

# HPCX ships an MPI that shadows the EFA installer's Open MPI.
RUN rm -rf /opt/hpcx /usr/local/mpi \
    && rm -f /etc/ld.so.conf.d/hpcx.conf \
    && ldconfig
ENV OPAL_PREFIX=

# --- GDRCopy -----------------------------------------------------------------
# NVSHMEM uses GDRCopy for low-latency host-to-GPU-memory writes on the proxy
# path, which is exactly the path the EFA patch routes DeepEP's RDMA through.
ARG GDRCOPY_PREFIX=/opt/gdrcopy
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix="${GDRCOPY_PREFIX}" install \
    && rm -rf /tmp/gdrcopy

ENV LD_LIBRARY_PATH="${GDRCOPY_PREFIX}/lib:${LD_LIBRARY_PATH}"
# The CUDA driver stubs must be on LIBRARY_PATH or DeepEP's link step fails with
# "/usr/bin/ld: cannot find -lcuda". nvidia/cuda:*-devel (the base of
# deepep-benchmark's own Dockerfile) sets LIBRARY_PATH=/usr/local/cuda/lib64/stubs
# for exactly this reason; the vLLM image sets LIBRARY_PATH to nothing. The real
# libcuda is not in the image at all -- the nvidia container runtime injects it at
# `docker run`, which does not happen during `docker build`.
# Stubs go on LIBRARY_PATH (link time) ONLY, never LD_LIBRARY_PATH: at runtime the
# stub would shadow the injected driver and every CUDA call would fail.
ENV LIBRARY_PATH="${GDRCOPY_PREFIX}/lib:/usr/local/cuda/lib64/stubs"
ENV CPATH="${GDRCOPY_PREFIX}/include"
ENV PATH="${GDRCOPY_PREFIX}/bin:${PATH}"

# --- EFA installer -----------------------------------------------------------
# --skip-kmod: the kernel module comes from the host, not the container.
# --no-verify: verification wants a working fabric, unavailable at build time.
# Installs libfabric (the transport NVSHMEM will use) and aws-ofi-nccl.
RUN cd /tmp \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && if printf '%s\n' "1.47.0" "${EFA_INSTALLER_VERSION}" | sort -V | head -n1 | grep -qx "${EFA_INSTALLER_VERSION}"; then \
        ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify; \
    else \
        ./efa_installer.sh --disable-build-ngc --disable-ngc -y --skip-kmod --skip-limit-conf --no-verify; \
    fi \
    && ldconfig \
    && rm -rf /tmp/aws-efa-installer /tmp/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
ENV PATH=/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:${PATH}

# --- NVSHMEM 3.7.0 + DeepEP v1 (EFA-patched) ---------------------------------
# setup_deepep_efa.sh is bind-mounted from the deepep-benchmark directory rather
# than copied, so this image and the micro-benchmark image are built from one
# script. It installs prebuilt NVSHMEM (sha256-verified), clones DeepEP at the
# only supported commit, applies the embedded EFA patch that swaps IBGDA device
# calls for NVSHMEM host-proxy QP APIs, and pip-installs deep_ep.
#
# The base image ships nvidia-nvshmem-cu13 3.4.5 via pip. It is BELOW the 3.7.0
# floor DeepEP v1 requires and would shadow the standalone install at runtime,
# so it is removed in the same layer.
#
# CPATH: the base image's /usr/local/cuda/include is a PARTIAL toolkit -- it has
# cublas and curand but not cusparse, cusolver or cufft. DeepEP pulls in torch's
# ATen/cuda/CUDAContextLight.h, which includes cusparse.h, so the internode.cu
# compile fails with "fatal error: cusparse.h: No such file or directory". The
# full set of math-library headers ships in the pip nvidia/cu13 tree, so point
# CPATH at it. Do NOT drop this without re-verifying the compile.
ARG DEEPEP_PREFIX=/opt/amazon/deepep
RUN --mount=type=bind,source=setup_deepep_efa.sh,target=/tmp/setup_deepep_efa.sh \
    CPATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include:${CPATH}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    /tmp/setup_deepep_efa.sh \
        --nvshmem-version "${NVSHMEM_VERSION}" \
        --deepep-commit "${DEEPEP_COMMIT}" \
        --deepep-prefix "${DEEPEP_PREFIX}" \
        --gdrcopy-home "${GDRCOPY_PREFIX}" \
        --gen-ldconfig \
    && { pip3 uninstall -y nvidia-nvshmem-cu13 nvidia-nvshmem-cu12 nvidia-nvshmem || true; } \
    && LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}:${LD_LIBRARY_PATH}" \
       python3 -c "import torch; from deep_ep import Buffer; print('deep_ep built OK')"

ENV LD_LIBRARY_PATH=/opt/amazon/nvshmem/lib:${LD_LIBRARY_PATH}
ENV PATH=/opt/amazon/nvshmem/bin:${PATH}

# --- NVSHMEM libfabric transport selection -----------------------------------
# Baked in so every rank inherits them; DeepEP's internode kernels target IBGDA,
# which EFA does not provide, and without NVSHMEM_REMOTE_TRANSPORT=libfabric
# NVSHMEM falls back to IBGDA and dies with "DeepEP error: CPU recv timeout".
ENV FI_PROVIDER=efa
ENV NVSHMEM_REMOTE_TRANSPORT=libfabric
ENV NVSHMEM_LIBFABRIC_PROVIDER=efa
ENV NVSHMEM_NETDEVS_POLICY=EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE
ENV NVIDIA_GDRCOPY=enabled

# --- smoke tests -------------------------------------------------------------
# Fail the build rather than a multi-node job. The nvshmem version and the
# absence of the stale pip wheel are both asserted because either would produce
# a runtime failure that looks like a fabric problem.
#
# LD_LIBRARY_PATH is prefixed with the CUDA compat dir for this layer only:
# deep_ep_cpp links libcuda.so.1 and the host driver is not injected during
# `docker build`. It is deliberately NOT part of the image's ENV.
RUN LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}:${LD_LIBRARY_PATH}"; export LD_LIBRARY_PATH; \
    echo "=== vLLM ===" \
    && python3 -c "import vllm; print(f'vLLM {vllm.__version__}')" \
    && echo "=== torch/CUDA ===" \
    && python3 -c "import torch; assert torch.version.cuda.startswith('13'), torch.version.cuda; print(torch.__version__, torch.version.cuda)" \
    && echo "=== deep_ep imports ===" \
    && python3 -c "import torch; from deep_ep import Buffer, Config; print('deep_ep OK')" \
    && echo "=== vLLM sees deep_ep ===" \
    && python3 -c "from vllm.utils.import_utils import has_deep_ep; assert has_deep_ep(); print('has_deep_ep() = True')" \
    && echo "=== deep_ep links standalone NVSHMEM 3.7.0, not the pip wheel ===" \
    && python3 -c "\
import importlib.util, subprocess; \
spec = importlib.util.find_spec('deep_ep_cpp'); \
assert spec and spec.origin, 'deep_ep_cpp extension not found'; \
out = subprocess.check_output(['ldd', spec.origin], text=True); \
nv = [l.strip() for l in out.splitlines() if 'libnvshmem' in l]; \
assert nv, 'deep_ep_cpp does not link NVSHMEM at all:\n' + out; \
assert all('=> /opt/amazon/nvshmem/lib/' in l for l in nv), \
    'NVSHMEM resolved outside the standalone install: ' + str(nv); \
print('deep_ep_cpp NVSHMEM link OK: ' + '; '.join(nv))" \
    && echo "=== stale nvshmem pip wheel absent ===" \
    && python3 -c "\
import glob, importlib.util, subprocess; \
libs = glob.glob('/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/**/*.so*', recursive=True); \
assert not libs, 'stale pip NVSHMEM libs would shadow the standalone install: ' + str(libs); \
out = subprocess.check_output(['pip3', 'list'], text=True); \
assert 'nvshmem' not in out.lower(), 'nvshmem pip package still installed'; \
print('no pip NVSHMEM (pip leaves the empty dir behind; only .so files would shadow)')" \
    && echo "=== NVSHMEM >= 3.7.0 ===" \
    && python3 -c "\
import glob, re, sys; \
libs = glob.glob('/opt/amazon/nvshmem/lib/libnvshmem_host.so.*.*.*'); \
assert libs, 'no versioned libnvshmem_host.so found'; \
ver = max(tuple(int(x) for x in re.search(r'so\.(\d+)\.(\d+)\.(\d+)\$', l).groups()) for l in libs); \
assert ver >= (3, 7, 0), 'NVSHMEM %s is below the 3.7.0 floor DeepEP v1 requires' % (ver,); \
print('NVSHMEM %d.%d.%d OK' % ver)"
RUN echo "=== libfabric transport present ===" \
    && ls /opt/amazon/nvshmem/lib/nvshmem_transport_libfabric*.so \
    && ldconfig -p | grep -q libfabric \
    && fi_info --version | head -1 \
    && echo "=== All smoke tests passed ==="

WORKDIR /vllm-workspace
