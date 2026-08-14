#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# Smoke the EFA + DeepEP-V2 GDAKI substrate in the built image BEFORE loading a model. Fails loud.
# (Single-node static checks — no rendezvous. The cross-node byte-moving proof is run-kernel-test.sh.)
set -euo pipefail
IMG="${1:?usage: verify-image.sh <image>}"
docker run --rm --gpus all "${IMG}" bash -lc '
  set -euo pipefail
  echo "== fi_info efa (newer libfabric build must still see efa-direct) =="
  /opt/libfabric-gdaki/bin/fi_info -p efa 2>/dev/null | grep -q "efa" || /opt/amazon/efa/bin/fi_info -p efa | grep -q "fabric: efa-direct" || { echo "FAIL: no efa provider"; exit 1; }
  echo "== rdma-core comp-cntr verbs present (PR#1701) =="; nm -D /opt/rdma-core-gdaki/lib/libefa.so.1 | grep -qi comp_cntr || { echo "FAIL: no comp_cntr verbs"; exit 1; }
  echo "== single libnccl 2.30.4 wins =="; ldconfig -p | grep "libnccl.so.2 " | head -1 | grep -q "nvidia/nccl" || { echo "FAIL: system libnccl shadows pip"; exit 1; }
  echo "== GIN plugin symbol =="; nm -D /opt/aws-ofi-nccl-gdaki/lib/libnccl-net-ofi.so | grep -q ncclGinPlugin || { echo "FAIL: no ncclGinPlugin"; exit 1; }
  echo "== GDAKI compiled into the plugin =="; strings /opt/aws-ofi-nccl-gdaki/lib/libnccl-net-ofi.so | grep -qi gdaki || { echo "FAIL: no gdaki strings"; exit 1; }
  echo "== hw-counter tristate param baked (PR#1311) =="; strings /opt/aws-ofi-nccl-gdaki/lib/libnccl-net-ofi.so | grep -q GDAKI_EFA_HW_COUNTER || { echo "FAIL: no GDAKI_EFA_HW_COUNTER param"; exit 1; }
  echo "== DeepEP ElasticBuffer import + V13 shim marker =="; grep -q V13_HOST_UC_SHIM /opt/DeepEP/csrc/elastic/buffer.hpp || { echo "FAIL: V13 shim missing"; exit 1; }
  python3 -c "import deep_ep; assert hasattr(deep_ep,\"ElasticBuffer\"), \"no ElasticBuffer\"; print(\"ElasticBuffer OK\")" 2>/dev/null || echo "(note: deep_ep _C.so builds in-pod on first boot — ElasticBuffer import is verified there)"
  echo "ALL STATIC CHECKS PASS"
'
