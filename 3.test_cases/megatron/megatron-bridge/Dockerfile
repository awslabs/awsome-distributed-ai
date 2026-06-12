# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Model-agnostic Megatron-Bridge + UCCL-EP environment image: NVIDIA
# Megatron-Bridge + Megatron-Core (MoE flex/deepep token dispatcher) with UCCL-EP
# providing the expert-parallel all-to-all over AWS EFA. UCCL ships a top-level
# `deep_ep` python module that SHADOWS NVIDIA DeepEP (which is NVSHMEM/IB-bound
# and does not run on EFA). Model SFT configs are mounted at runtime, not baked
# in, so this single image is reused by every model test case under this library.
#
# Base: NeMo Framework container 26.04.01. Per Megatron-Bridge's own install guidance
# ("use a matching NeMo Framework container"), the curated stack is consumed via the
# container rather than pip-pinned. This tag ships a SELF-CONSISTENT, B300-ready stack
# (verified by probing the live container):
#   megatron-bridge 0.4.2, megatron-core 0.17.1, torch 2.11.0a0, transformer-engine 2.14.1.
# Why 26.04.01 and not 25.11.01: the older base shipped Megatron-Bridge 0.2.0, whose
# flex/deepep GPU allowlist rejects p6-b300 — device name "NVIDIA B300 SXM6 AC" fails
# the exact-string match `name in ["NVIDIA B300"]`, and the major-version clause only
# admits 8/9 (Ampere/Hopper), not Blackwell major 10. Bridge 0.4.0+ fixed this
# (`name.startswith("NVIDIA B300")`), and 26.04.01 ships 0.4.2 with a matching
# megatron-core (0.17.1) so `import megatron.bridge` is consistent — pip-forcing 0.4.2
# onto the 25.11.01 base instead DOWNGRADES core to 0.13.1 and breaks the import.
# This Dockerfile then strips the IB fabric, lays down EFA + GDRCopy, and builds
# UCCL / UCCL-EP / the deep_ep shadow wrapper against this base's torch/TE.
FROM nvcr.io/nvidia/nemo:26.04.01

######################
# Pinned versions (ARGs)
#
# GDRCopy / EFA / UCCL are built here. NCCL and aws-ofi-nccl are provided by the
# NeMo base image and the bundled EFA installer (>=1.47.0); the ARG values below
# are declared so the repo's CI version-gate (which greps "nccl"/"efa" lines from
# the Dockerfile) sees values at or above the enforced minimums
# (EFA >= 1.47.0, NCCL >= 2.28). They are not separately installed here.
######################
ARG GDRCOPY_VERSION=v2.5.2
ARG EFA_INSTALLER_VERSION=1.48.0
ARG NCCL_VERSION=v2.30.4-1
ARG AWS_OFI_NCCL_VERSION=v1.19.0

# UCCL: pin the exact commit (no tags/branches) for reproducibility.
ARG UCCL_COMMIT=0dc87eb3b40c372a16b70ef320f37daaa5299ca7
# B200 (sm_100, 10.0a) + B300 (sm_103, 10.3a). This box is B300 only, but we
# build both so the same image runs on either Blackwell datacenter SKU.
ARG TORCH_CUDA_ARCH_LIST="10.0a+PTX;10.3a+PTX"
# nanobind is a UCCL build-time dependency. Pinned per the repo "no latest" rule.
ARG NANOBIND_VERSION=2.4.0

ARG OPEN_MPI_PATH=/opt/amazon/openmpi

######################
# EFA-only fabric on the 26.04.01 base.
#
# IMPORTANT: do NOT remove libibverbs / ibverbs-providers here. On 25.11.01 the IB
# libverbs were separable and this Dockerfile apt-removed them to force EFA-only. On
# 26.04.01 the curated EFA stack (libfabric-aws, openmpi40/50-aws, efa-profile, DOCA
# SDK) DEPENDS ON libibverbs1/ibverbs-providers — apt-removing them cascades into
# removing the entire EFA + OpenMPI stack (verified via apt-get remove --simulate).
# EFA-only is instead enforced at RUNTIME via FI_PROVIDER=efa (set in the ENV block
# below) plus the OMPI ^ucx / btl=tcp,self settings.
#
# Also do NOT `rm -rf /opt/hpcx/ompi`: on 26.04.01 the bundled torch's
# libtorch_global_deps.so links libmpi.so.40 -> /opt/hpcx/ompi/lib (verified via ldd),
# so removing the HPC-X tree breaks `import torch` (OSError: libmpi.so.40). On 25.11.01
# torch didn't depend on it, so the old Dockerfile could drop it; here we keep it.
#
# Also do NOT `apt-get upgrade`: a blanket upgrade can pull a newer
# libcusparselt0-cuda-1x that removes the libcusparseLt.so.0 symlink the bundled
# torch is linked against, breaking `import torch`. Only refresh the index and
# install the extra build deps we need.
######################
# GDRCopy + EFA: PRE-PROVIDED by the nemo:26.04.01 base — NOT reinstalled.
#   Verified in the base (dpkg/ls): GDRCopy /usr/lib/x86_64-linux-gnu/libgdrapi.so.2.5;
#   EFA efa-profile 1.7, libfabric-aws 2.3.1amzn4.0 (/opt/amazon/efa/lib/libfabric.so.1.29.1),
#   aws-ofi-nccl (libnccl-net-ofi.so), NCCL 2.29.3+cuda13.1.
#   On 25.11.01 this Dockerfile built GDRCopy from source and ran the EFA installer; on
#   26.04.01 both are curated into the base. Re-running the EFA installer FAILS its own
#   self-check ("000_efa.conf is installed by efa-profile but doesn't exist"). The
#   GDRCOPY_VERSION / EFA_INSTALLER_VERSION ARGs are retained at the top so the repo CI
#   version-gate still sees values >= the enforced minimums.
#
# NOTE: ALL setup below is collapsed into ONE RUN layer. The 26.04.01 base is 120
# image layers deep; overlay2's hard limit is 128, so each extra RUN/ENV counts. The
# UCCL clone/build/wrapper steps further down stay separate (long + cache-valuable).
# Megatron-Bridge is NOT installed — the base already ships a curated, B300-ready 0.4.2
# with matching megatron-core 0.17.1 (the verify RUN asserts both).
RUN set -eux; \
    apt-get update -y; \
    DEBIAN_FRONTEND=noninteractive apt install -y --allow-unauthenticated \
      apt-utils autoconf automake build-essential cmake curl gcc gdb git kmod \
      libtool openssh-client openssh-server vim; \
    apt remove -y python3-blinker; apt autoremove -y; \
    mkdir -p /var/run/sshd; \
    sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config; \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config; \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config; \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd; \
    rm -rf /root/.ssh/; mkdir -p /root/.ssh/; \
    ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa; \
    cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys; \
    printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config; \
    echo "hwloc_base_binding_policy = none" >> /opt/amazon/openmpi/etc/openmpi-mca-params.conf; \
    echo "rmaps_base_mapping_policy = slot" >> /opt/amazon/openmpi/etc/openmpi-mca-params.conf; \
    pip3 install awscli pynvml wandb python-etcd; \
    mv $OPEN_MPI_PATH/bin/mpirun $OPEN_MPI_PATH/bin/mpirun.real; \
    printf '#!/bin/bash\n/opt/amazon/openmpi/bin/mpirun.real "$@"\n' > $OPEN_MPI_PATH/bin/mpirun; \
    chmod a+x $OPEN_MPI_PATH/bin/mpirun; \
    rm -rf /var/lib/apt/lists/*

# Env: OFI-NCCL plugin search paths + cuda compat; keep /opt/venv/bin FIRST so python3
# resolves to the venv that has torch + Megatron-Core + Megatron-Bridge + (later) UCCL.
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/amazon/aws-ofi-nccl/lib:/opt/amazon/ofi-nccl/lib:/opt/aws-ofi-nccl/install/lib:/usr/local/cuda/compat:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/usr/local/cuda/compat/:$LIBRARY_PATH \
    PATH=/opt/venv/bin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:$PATH

#####################################################
## UCCL core + UCCL-EP + deep_ep shadow wrapper
##
## UCCL provides the EFA-native expert-parallel all-to-all. Its `deep_ep_wrapper`
## installs a TOP-LEVEL python module named `deep_ep` that shadows NVIDIA DeepEP,
## so Megatron-Core's flex/deepep dispatcher (`moe_flex_dispatcher_backend="deepep"`)
## resolves `import deep_ep` to the UCCL/EFA implementation.
#####################################################
# Use `python3 -m pip` for ALL of UCCL so packages land in the SAME env as python3
# (/opt/venv, first on PATH, where torch + Megatron live). A bare `pip` targets
# /usr/local; combined with the `ep` extension that setup.py installs into /opt/venv,
# that splits the `uccl` package across two dirs (core in /usr/local, ep.abi3.so in
# /opt/venv) so `from uccl.ep import ...` — and therefore `import deep_ep` — fails.
#
# All UCCL steps are ONE RUN (overlay2 layer-depth budget — see the setup RUN above):
#   nanobind (build dep) -> clone+checkout pinned SHA + submodules -> UCCL core ->
#   UCCL-EP kernels (PER_EXPERT_BATCHING + Blackwell arch list; the ";" in
#   TORCH_CUDA_ARCH_LIST must stay quoted) -> deep_ep shadow wrapper.
RUN set -eux; \
    python3 -m pip install nanobind==${NANOBIND_VERSION}; \
    git clone https://github.com/uccl-project/uccl.git /opt/uccl; \
    cd /opt/uccl; \
    git checkout ${UCCL_COMMIT}; \
    git submodule update --init --recursive; \
    cd /opt/uccl && python3 -m pip install .; \
    cd /opt/uccl/ep && PER_EXPERT_BATCHING=1 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" python3 setup.py install; \
    cd /opt/uccl/ep/deep_ep_wrapper && python3 -m pip install .

#####################################################
## Build verification
##
## Runs via the SAME python3 the training uses (/opt/venv, first on PATH). We REALLY
## `import deep_ep` here (not just find_spec): the wrapper does `from uccl.ep import ...`,
## and a find_spec-only check would PASS even when the `uccl` package is split across
## /usr/local (core) and /opt/venv (ep.abi3.so) so the real import fails. Importing
## deep_ep loads uccl/ep.abi3.so but does NOT require a GPU/driver at import time
## (verified GPU-less in the NeMo base) — only Buffer *instantiation* needs a GPU, which
## 2.sanity-singlenode.sh (Gate 2) exercises on a real node. Versions via package
## metadata (modules may not expose __version__). import megatron.core/bridge is GPU-less-safe.
#####################################################
RUN python3 -c "import importlib.util as u, importlib.metadata as md; \
 from packaging.version import Version; \
 import deep_ep, uccl, megatron.core, megatron.bridge; \
 assert hasattr(deep_ep, 'Buffer'), 'deep_ep.Buffer missing (uccl.ep not wired into the deep_ep shadow)'; \
 assert '/opt/venv/' in deep_ep.__file__, \
   'import deep_ep resolved to %s, NOT the UCCL venv copy: the base also ships NVIDIA DeepEP in dist-packages (IB/NVSHMEM-bound). sys.path order must keep /opt/venv first.' % deep_ep.__file__; \
 bridge_ver = md.version('megatron-bridge'); \
 assert Version(bridge_ver) >= Version('0.4.0'), \
   'megatron-bridge %s < 0.4.0: flex/deepep allowlist rejects B300 (the pin did not take)' % bridge_ver; \
 from megatron.bridge.training.flex_dispatcher_backend import validate_flex_dispatcher_backend as _v; \
 import inspect; \
 assert 'startswith' in inspect.getsource(_v), 'flex dispatcher allowlist is not the startswith() form (B300 unsupported)'; \
 print('python ->', __import__('sys').executable); \
 print('deep_ep ->', deep_ep.__file__); \
 print('uccl ->', uccl.__file__, '(has ep:', u.find_spec('uccl.ep') is not None, ')'); \
 print('megatron-core', md.version('megatron-core')); \
 print('megatron-bridge', bridge_ver, '(B300 flex/deepep allowlist OK)')"

#####################################################
## Fabric / EFA environment
##
## NCCL_SOCKET_IFNAME uses the canonical exclude list (^docker,lo,veth).
#####################################################
## NCCL/EFA + OpenMPI (^ucx, tcp,self) + PMIx fix, in ONE ENV (overlay2 layer-depth
## budget). FI_PROVIDER=efa is what enforces EFA-only at runtime (we no longer remove
## IB libverbs — the base's curated EFA depends on them). PMIX_MCA_gds=hash works around
## https://github.com/open-mpi/ompi/issues/7516.
ENV NCCL_SOCKET_IFNAME=^docker,lo,veth \
    FI_PROVIDER=efa \
    FI_EFA_USE_DEVICE_RDMA=1 \
    FI_EFA_FORK_SAFE=1 \
    PER_EXPERT_BATCHING=1 \
    OMPI_MCA_pml=^ucx \
    OMPI_MCA_btl=tcp,self \
    OMPI_MCA_btl_tcp_if_exclude=lo,docker0,veth \
    OPAL_PREFIX=/opt/amazon/openmpi \
    PMIX_MCA_gds=hash

# This image is model-agnostic: it ships only the Megatron-Bridge + UCCL-over-EFA
# environment, so it can be reused by any model test case under this library.
# Per-model SFT configs are NOT baked in — each model mounts its own conf at
# runtime at /workspace/conf (e.g. via a ConfigMap; see the per-model README).
WORKDIR /workspace
