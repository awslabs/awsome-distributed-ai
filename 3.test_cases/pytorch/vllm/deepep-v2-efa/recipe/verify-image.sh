#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# Smoke the EFA + DeepEP-V2 substrate in the built image BEFORE loading a model. Fails loud.
set -euo pipefail
IMG="${1:?usage: verify-image.sh <image>}"
docker run --rm --gpus all "${IMG}" bash -lc '
  set -euo pipefail
  echo "== fi_info efa =="; /opt/amazon/efa/bin/fi_info -p efa | grep -q "fabric: efa-direct" || { echo "FAIL: no efa-direct"; exit 1; }
  echo "== single libnccl 2.30.4 wins =="; ldconfig -p | grep "libnccl.so.2 " | head -1 | grep -q "nvidia/nccl" || { echo "FAIL: system libnccl shadows pip"; exit 1; }
  echo "== GIN plugin symbol =="; nm -D /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep -q ncclGinPlugin || { echo "FAIL: no ncclGinPlugin"; exit 1; }
  echo "== DeepEP ElasticBuffer import =="; python3 -c "import deep_ep; assert hasattr(deep_ep,\"ElasticBuffer\"), \"no ElasticBuffer\"; print(\"ElasticBuffer OK\")"
  echo "ALL CHECKS PASS"
'
