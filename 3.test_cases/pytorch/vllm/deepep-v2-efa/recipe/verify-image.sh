#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# Smoke the EFA + DeepEP-V2 substrate in the built image BEFORE loading a model. Fails loud.
# Static image check: asserts exactly what the image stages. The DeepEP _C.so is built
# IN-POD on first boot (build_deepep.sh — needs a live CUDA context), so `import deep_ep`
# is deliberately NOT asserted here; the staged source + the transport substrate are.
set -euo pipefail
IMG="${1:?usage: verify-image.sh <image>}"

# EFA device mapping: fi_info -p efa only resolves with the device visible in the
# container. On an EFA host, pass /dev/infiniband through; elsewhere fall back to a
# provider-compiled-in check (fi_info -l) and say so.
DEV_ARGS=()
HAVE_EFA_DEV=0
if [ -d /dev/infiniband ]; then
  DEV_ARGS=(-v /dev/infiniband:/dev/infiniband --device=/dev/infiniband)
  HAVE_EFA_DEV=1
fi

docker run --rm --gpus all "${DEV_ARGS[@]}" -e HAVE_EFA_DEV="${HAVE_EFA_DEV}" "${IMG}" bash -lc '
  set -euo pipefail
  if [ "${HAVE_EFA_DEV}" = "1" ]; then
    echo "== fi_info efa (live fabric) =="
    /opt/amazon/efa/bin/fi_info -p efa | grep -q "fabric: efa-direct" || { echo "FAIL: no efa-direct"; exit 1; }
  else
    echo "== fi_info efa (no /dev/infiniband on this host — checking provider is compiled in only) =="
    /opt/amazon/efa/bin/fi_info -l | grep -qi "efa" || { echo "FAIL: efa provider not in libfabric"; exit 1; }
    echo "   (run this script on an EFA host for the live efa-direct fabric check)"
  fi
  echo "== single libnccl 2.30.4 wins (path + GIN/LSA symbol — the wheel downgrade lands in the SAME dir, so path alone cannot catch it) =="
  NCCL_SO=$(ldconfig -p | grep "libnccl.so.2 " | head -1 | awk "{print \$NF}")
  echo "$NCCL_SO" | grep -q "nvidia/nccl" || { echo "FAIL: system libnccl shadows pip ($NCCL_SO)"; exit 1; }
  nm -D "$NCCL_SO" | grep -q ncclGetLsaDevicePointer || { echo "FAIL: $NCCL_SO lacks GIN/LSA symbols (2.28.x downgrade — see Dockerfile Layer 5b)"; exit 1; }
  echo "== GIN plugin symbol =="; nm -D /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep -q ncclGinPlugin || { echo "FAIL: no ncclGinPlugin"; exit 1; }
  echo "== DeepEP-V2 source staged (built in-pod on first boot; import is asserted there, not here) =="
  test -f /opt/DeepEP/deep_ep/buffers/elastic.py || { echo "FAIL: /opt/DeepEP not staged"; exit 1; }
  grep -q "class ElasticBuffer" /opt/DeepEP/deep_ep/buffers/elastic.py || { echo "FAIL: staged DeepEP has no ElasticBuffer (V1 source?)"; exit 1; }
  test -f /opt/DeepEP/tests/elastic/test_ep.py || { echo "FAIL: kernel-smoke test not staged"; exit 1; }
  echo "ALL CHECKS PASS"
'
