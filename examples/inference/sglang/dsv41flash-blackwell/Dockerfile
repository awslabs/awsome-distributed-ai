# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# EFA-enabled SGLang image for the DeepSeek-V4.1-Flash PD recipe.
#
# DeepSeek-V4.1 support has not landed in a tagged SGLang release (checked up
# to v0.5.19); it only exists in the preview image lmsysorg/sglang:dev-dsv41.
# That image has no EFA/libfabric layer, so this Dockerfile adds the same AWS
# EFA installer layer as ../Dockerfile.efa on top of it. It lives here rather
# than reusing ../Dockerfile.efa because the base image differs from every
# other multi-node sample.
#
# dev-* tags are MUTABLE. Pin the base by digest before relying on the build:
#   docker buildx imagetools inspect lmsysorg/sglang:dev-dsv41   # -> sha256:...
#   BASE_IMAGE=lmsysorg/sglang@sha256:<digest> ./build-image.sh
# build-image.sh prints the resolved digest; record it here and in README.md.
ARG BASE_IMAGE=lmsysorg/sglang:dev-dsv41
FROM ${BASE_IMAGE}

ARG EFA_VERSION=1.50.0

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

# If the NIXL shipped in the base image reproduces the LIBFABRIC GPU-HMEM slow
# path documented in ../kimi2.6-h200-1p1d/README.md (seen with NIXL 1.2.0),
# pin a known-good wheel here and rebuild, e.g.:
# RUN pip install --no-deps --force-reinstall "nixl==<known-good>"
