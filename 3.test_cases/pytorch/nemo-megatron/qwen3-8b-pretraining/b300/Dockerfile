FROM nvcr.io/nvidia/nemo:26.02

# Install EFA prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    environment-modules \
    tcl \
    udev \
    ethtool \
    iproute2 \
    dmidecode \
    libevent-core-2.1-7t64 \
    libevent-pthreads-2.1-7t64 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install EFA (v1.47.0 - compatible with NeMo 26.02 base)
# Do NOT remove existing NCCL/hpcx - EFA installs on top
RUN cd /tmp && \
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.47.0.tar.gz && \
    tar -xf aws-efa-installer-1.47.0.tar.gz && \
    cd aws-efa-installer && \
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify && \
    cd / && rm -rf /tmp/aws-efa-installer*

# GDRCopy for GPU-direct RDMA
RUN cd /tmp && \
    git clone -b v2.4.4 https://github.com/NVIDIA/gdrcopy.git && \
    cd gdrcopy && \
    make -j$(nproc) lib lib_install && \
    cd / && rm -rf /tmp/gdrcopy

# Environment
ENV LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}"
ENV NCCL_TUNER_PLUGIN="/opt/amazon/ofi-nccl/lib/libnccl-tuner-aws-ofi.so"
ENV FI_PROVIDER=efa
ENV TORCH_COMPILE_DISABLE=1
ENV NCCL_PROTO=simple

WORKDIR /workspace
