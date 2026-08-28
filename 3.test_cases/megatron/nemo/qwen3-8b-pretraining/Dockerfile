FROM nvcr.io/nvidia/nemo:26.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libevent-core-2.1-7 libevent-pthreads-2.1-7 \
    ethtool iproute2 pciutils curl \
    && cd /tmp \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.47.0.tar.gz \
    && tar -xf aws-efa-installer-1.47.0.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify \
    && cd /tmp \
    && git clone -b v2.5.1 --depth 1 https://github.com/NVIDIA/gdrcopy.git \
    && cd gdrcopy && make -j$(nproc) lib lib_install \
    && cd / && rm -rf /tmp/* /var/lib/apt/lists/*

# Environment
ENV LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}"
ENV NCCL_TUNER_PLUGIN="/opt/amazon/ofi-nccl/lib/libnccl-tuner-aws-ofi.so"
ENV FI_PROVIDER=efa
ENV TORCH_COMPILE_DISABLE=1
ENV NCCL_PROTO=simple

WORKDIR /workspace
