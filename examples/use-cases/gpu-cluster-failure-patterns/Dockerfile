# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
FROM public.ecr.aws/hpc-cloud/nccl-tests@sha256:a5390d3f0eb50f3e5854085e2ae0fef90b9eb4f3b7486ed4cab0d47864809d4c
COPY requirements.txt /opt/aim344-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /opt/aim344-requirements.txt
COPY . /opt/aim344/
WORKDIR /opt/aim344
