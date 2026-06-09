FROM lmsysorg/sglang:dev-cu13

ARG EFA_VERSION=1.47.0

RUN apt-get update && apt-get install -y --no-install-recommends \
      pciutils environment-modules tcl \
      libnl-3-dev libnl-route-3-dev \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN cd /tmp \
  && curl -O https://s3-us-west-2.amazonaws.com/aws-efa-installer/aws-efa-installer-${EFA_VERSION}.tar.gz \
  && tar -xf aws-efa-installer-${EFA_VERSION}.tar.gz \
  && cd aws-efa-installer \
  && ./efa_installer.sh -y --skip-kmod -g \
  && rm -rf /tmp/aws-efa-installer*

ENV PATH="/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH"

ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1
