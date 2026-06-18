FROM nvcr.io/nvidia/nemo:25.07

# EFA + NCCL OFI plugin for multi-node communication
# The NeMo base image includes PyTorch 2.8.0, CUDA 12.9, Megatron-Core 0.13.1, TE 2.5
# We add EFA support for AWS inter-node networking via GDRDMA

ARG EFA_INSTALLER_VERSION=1.48.0
ARG NCCL_VERSION=v2.30.4-1
ARG GDRCOPY_VERSION=v2.5.2

# Remove existing NCCL/MPI to avoid conflicts
RUN apt-get update -y && \
    apt-get remove -y --allow-change-held-packages \
    ibverbs-utils libibverbs-dev libibverbs1 \
    libmlx5-1 libnccl2 libnccl-dev || true && \
    rm -rf /opt/hpcx /etc/ld.so.conf.d/hpcx.conf && ldconfig

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential cmake curl git kmod libtool \
    openssh-client openssh-server pkg-config vim wget

# SSH config for multi-node
RUN mkdir -p /var/run/sshd && \
    sed -i 's/[ #]\(.*StrictHostKeyChecking\).*/\1 no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes\).*/\1 no/g' /etc/ssh/sshd_config

# GDRCopy for GPU-direct RDMA
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy && make prefix=/opt/gdrcopy install

# EFA installer (provides libfabric + OFI NCCL plugin)
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf $HOME/aws-efa-installer

# Build NCCL from source (sm_90 for H200)
RUN git clone -b ${NCCL_VERSION} https://github.com/NVIDIA/nccl.git /opt/nccl \
    && cd /opt/nccl \
    && make -j $(nproc) src.build CUDA_HOME=/usr/local/cuda \
    NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

ENV LD_LIBRARY_PATH=/opt/gdrcopy/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/openmpi/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/opt/gdrcopy/bin:$PATH

# EFA/NCCL environment
ENV FI_PROVIDER=efa
ENV NCCL_SOCKET_IFNAME=^docker,lo,veth
ENV NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-tuner-aws-ofi.so
ENV LD_PRELOAD=/opt/nccl/build/lib/libnccl.so
ENV TORCH_COMPILE_DISABLE=1

# OpenMPI settings
ENV OMPI_MCA_pml=^ucx \
    OMPI_MCA_btl=tcp,self \
    OMPI_MCA_btl_tcp_if_exclude=lo,docker0,veth_def_agent \
    OPAL_PREFIX=/opt/amazon/openmpi \
    PMIX_MCA_gds=hash

RUN rm -rf /var/lib/apt/lists/* /tmp/gdrcopy
