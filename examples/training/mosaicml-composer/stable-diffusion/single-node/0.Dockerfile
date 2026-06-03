# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

ARG PYTORCH_IMAGE

FROM ${PYTORCH_IMAGE}

ARG MOSAICML_VERSION
ARG PYTORCH_INDEX_URL

RUN git clone https://github.com/mosaicml/diffusion-benchmark.git /wd
RUN pip3 install -r /wd/requirements.txt
RUN pip3 install mosaicml==${MOSAICML_VERSION} --force
RUN pip3 install --pre torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL} --force
RUN pip3 uninstall transformer-engine -y
RUN pip3 install protobuf==3.20.3

WORKDIR /wd


