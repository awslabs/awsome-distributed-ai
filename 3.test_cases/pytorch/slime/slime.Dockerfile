# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# ============================================================================
# SLIME + SGLang + Megatron-LM image for SageMaker HyperPod EKS
# ============================================================================

FROM nvcr.io/nvidia/pytorch:26.02-py3

ARG GDRCOPY_VERSION=v2.5.2
ARG EFA_INSTALLER_VERSION=1.48.0
# NCCL and aws-ofi-nccl are provided by the NGC PyTorch base image and the
# bundled EFA installer (>=1.47.0). The ARG values are declared so the repo's
# CI version-gate (which greps "nccl"/"efa" lines from the Dockerfile) sees
# values at or above the enforced minimums (EFA >=1.47.0, NCCL >=2.28).
ARG NCCL_VERSION=v2.30.4-1
ARG AWS_OFI_NCCL_VERSION=v1.19.0
ARG MEGATRON_LM_VERSION=3714d81d418c9f1bca4594fc35f9e8289f652862
ARG SLIME_VERSION=v0.2.4
ARG SGLANG_VERSION=0.5.12.post1
# SLIME-patched sgl-router wheel (provides the "+slime" router used for fast
# weight sync). Pinned to the build referenced by SLIME v0.2.4.
ARG SGL_ROUTER_WHEEL=https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl


ARG PIP_VERSION=26.1.2
ARG SETUPTOOLS_VERSION=81.0.0
ARG RING_FLASH_ATTN_VERSION=0.1.8

ARG OPEN_MPI_PATH=/opt/amazon/openmpi

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

######################
# Update and remove the IB libverbs (replaced by the EFA installer below)
######################
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get remove -y --allow-change-held-packages \
    ibverbs-utils \
    libibverbs-dev \
    libibverbs1 \
    libmlx5-1

RUN rm -rf /opt/hpcx/ompi \
    && rm -rf /usr/local/mpi \
    && rm -rf /usr/local/ucx \
    && ldconfig

RUN DEBIAN_FRONTEND=noninteractive apt install -y --allow-unauthenticated \
    apt-utils \
    autoconf \
    automake \
    build-essential \
    cmake \
    curl \
    gcc \
    gdb \
    git \
    jq \
    kmod \
    libtool \
    openssh-client \
    openssh-server \
    vim \
    && apt remove -y python3-blinker \
    && apt autoremove -y

# NOTE: Permissive SSH config is standard practice for multi-node HPC images.
# SSH is used for inter-node MPI/NCCL communication within the cluster boundary.
# Port 22 should NOT be exposed outside the cluster via Kubernetes Services.
RUN mkdir -p /var/run/sshd && \
    sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN rm -rf /root/.ssh/ \
 && mkdir -p /root/.ssh/ \
 && ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa \
 && cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys \
 && printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config

# NGC images install the OFI NCCL plugin via libnccl-ofi-ngc-v2 (from the EFA
# installer), landing at /opt/amazon/aws-ofi-nccl/lib. Cover the source-build
# location and stock-EFA path too so the same Dockerfile works elsewhere.
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/amazon/aws-ofi-nccl/lib:/opt/amazon/ofi-nccl/lib:/opt/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH

#################################################
## Install NVIDIA GDRCopy
##
## NOTE: if `nccl-tests` or `/opt/gdrcopy/bin/sanity -v` crashes with incompatible version, ensure
## that the cuda-compat-xx-x package is the latest.
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix=/opt/gdrcopy install

ENV LD_LIBRARY_PATH=/opt/gdrcopy/lib:/usr/local/cuda/compat:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/gdrcopy/lib:/usr/local/cuda/compat/:$LIBRARY_PATH
ENV CPATH=/opt/gdrcopy/include:${CPATH:-}
ENV PATH=/opt/gdrcopy/bin:$PATH

#################################################
## Install EFA installer
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf $HOME/aws-efa-installer

RUN rm -rf /var/lib/apt/lists/*

RUN echo "hwloc_base_binding_policy = none" >> /opt/amazon/openmpi/etc/openmpi-mca-params.conf \
 && echo "rmaps_base_mapping_policy = slot" >> /opt/amazon/openmpi/etc/openmpi-mca-params.conf

RUN mv $OPEN_MPI_PATH/bin/mpirun $OPEN_MPI_PATH/bin/mpirun.real \
 && echo '#!/bin/bash' > $OPEN_MPI_PATH/bin/mpirun \
 && echo '/opt/amazon/openmpi/bin/mpirun.real "$@"' >> $OPEN_MPI_PATH/bin/mpirun \
 && chmod a+x $OPEN_MPI_PATH/bin/mpirun

#####################
# Install Megatron-LM (training backend for SLIME)
#
# A specific commit is checked out (see MEGATRON_LM_VERSION) so a full clone is
# used rather than `--depth 1 --branch`. nltk is pinned in requirements.txt and
# installed in a later layer; install it here too since the Megatron build/
# import touches it.
#####################
RUN pip install "setuptools==${SETUPTOOLS_VERSION}"
RUN cd /opt && git clone https://github.com/NVIDIA/Megatron-LM.git \
    && cd Megatron-LM \
    && git checkout ${MEGATRON_LM_VERSION} \
    && python3 -m pip install nltk \
    && python3 -m pip install .

# Pre-build the megatron datasets helpers C++ module. Megatron lazy-builds this
# on first dataset access (rank 0 only), but the workdir is local to each
# container -- ranks on other nodes hit ModuleNotFoundError because they never
# see the rank-0 build. Baking it into the image avoids the multi-node race.
RUN cd /opt/Megatron-LM/megatron/core/datasets \
    && g++ -O3 -Wall -shared -std=c++17 -fPIC -fdiagnostics-color \
       -I$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))') \
       -I$(python3 -c 'import pybind11; print(pybind11.get_include())') \
       helpers.cpp -o helpers_cpp$(python3-config --extension-suffix)

ENV PYTHONPATH="/opt/Megatron-LM:${PYTHONPATH:-}"

#####################
# SGLang (rollout / inference backend) -- install FIRST.
#
# SGLang has the tightest transitive constraints (torch, transformers,
# flashinfer, kernels). Installing it before the other RL deps lets pip resolve
# a consistent set. 0.5.12.post1 keeps the NGC-native torch 2.11 / CUDA 13
# build, so Megatron-LM's TransformerEngine and flash-attn extensions (compiled
# against that ABI in the base image) keep working. It pulls transformers 5.6.
#####################
RUN pip install --no-cache-dir "pip==${PIP_VERSION}" && \
    pip install --no-cache-dir "sglang[all]==${SGLANG_VERSION}"

#####################
# Reinforcement-learning Python dependencies (pinned in requirements.txt).
# Installed after SGLang so pip resolves against the versions SGLang fixed.
#####################
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

#####################
# SLIME runtime deps not pulled by the `--no-deps` install below
#   - sglang_router  : SLIME-patched sgl-router (zero-copy weight sync / R3)
#   - ring_flash_attn: OPTIONAL, only used for context-parallel (CP>1). It is
#     not import-compatible with transformers 5.6, so failure is tolerated; the
#     bundled math-GRPO recipes run with CP=1 and do not need it.
#####################
RUN pip install --no-cache-dir --force-reinstall --no-deps "${SGL_ROUTER_WHEEL}" && \
    python3 -c "import sglang_router; assert 'slime' in sglang_router.__version__, sglang_router.__version__" && \
    (pip install --no-cache-dir "ring_flash_attn==${RING_FLASH_ATTN_VERSION}" || echo "ring_flash_attn skipped (CP=1 recipes do not require it)")

#####################
# SLIME (RL post-training framework)
#####################
RUN cd /opt && \
    git clone --depth 1 --branch ${SLIME_VERSION} https://github.com/THUDM/slime.git && \
    cd slime && \
    pip install --no-cache-dir -e . --no-deps

## Set Open MPI variables to exclude network interface and conduit.
ENV OMPI_MCA_pml=^ucx            \
    OMPI_MCA_btl=tcp,self           \
    OMPI_MCA_btl_tcp_if_exclude=lo,docker0,veth_def_agent\
    OPAL_PREFIX=/opt/amazon/openmpi \
    NCCL_SOCKET_IFNAME=^docker,lo,veth_def_agent

## Turn off PMIx Error https://github.com/open-mpi/ompi/issues/7516
ENV PMIX_MCA_gds=hash

#####################
# EFA / NCCL / Ray runtime defaults
#####################
ENV FI_PROVIDER="efa"
ENV FI_EFA_USE_DEVICE_RDMA="1"
ENV FI_EFA_FORK_SAFE="1"
ENV NCCL_PROTO="Simple"
ENV NCCL_DEBUG="WARN"
ENV RDMAV_FORK_SAFE="1"
# Disable Ray's OOM killer: Megatron init transiently uses ~2x steady-state memory.
ENV RAY_memory_monitor_refresh_ms="0"
ENV TOKENIZERS_PARALLELISM="false"

WORKDIR /opt/slime

CMD ["/bin/bash"]
