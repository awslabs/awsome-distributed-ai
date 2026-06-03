#!/usr/bin/env bash
set -ex

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Remove old sqsh file if exists
if [ -f pytorch.sqsh ] ; then
    rm pytorch.sqsh
fi

docker build -t pytorch-ddp -f ../Dockerfile ..
enroot import -o pytorch.sqsh dockerd://pytorch-ddp:latest