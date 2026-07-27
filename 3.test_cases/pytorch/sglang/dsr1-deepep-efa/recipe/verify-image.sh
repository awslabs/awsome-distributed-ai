#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Single-node sanity checks on the built image. No multi-node setup needed.
# Run this before any 2-node work — it catches the shadowing and version-collision
# failure modes described in the README, which otherwise surface as confusing
# NVSHMEM errors much later.
#
# Usage:
#   source setup/env_vars
#   recipe/verify-image.sh

set -euo pipefail

: "${IMAGE_URI:?source setup/env_vars first}"

echo "==> Checking image ${IMAGE_URI}"

docker run --rm --gpus all \
    --device /dev/infiniband --device /dev/gdrdrv \
    --ulimit memlock=-1 \
    --entrypoint bash "$IMAGE_URI" -c '
set -euo pipefail
fail=0

echo "--- EFA provider visible to libfabric ---"
if fi_info -p efa >/dev/null 2>&1; then
    echo "OK: fi_info -p efa found $(fi_info -p efa | grep -c "provider: efa") endpoints"
else
    echo "FAIL: fi_info -p efa found no EFA provider"; fail=1
fi

echo "--- deep_ep resolves to the EFA build in the venv ---"
python3 - <<PY || fail=1
import os, sys
import deep_ep
p = os.path.dirname(deep_ep.__file__)
print("deep_ep ->", p)
if not p.startswith("/opt/deepep-venv"):
    print("FAIL: deep_ep is NOT the venv EFA build — a stock IBGDA copy is shadowing it")
    sys.exit(1)
print("OK")
PY

echo "--- NVSHMEM host library is v3.7.0 everywhere ---"
# Any libnvshmem_host.so.3 that resolves to something other than 3.7.0 will
# abort DeepEP at nvshmem init with a device/host version mismatch.
for lib in $(find / -name "libnvshmem_host.so.3" -not -path "/proc/*" 2>/dev/null); do
    # No early-exiting stage in this pipeline: under `set -o pipefail`, a
    # `grep -m1` would SIGPIPE `strings` and make the whole pipeline look failed.
    ver=$(strings "$lib" | grep -o "NVSHMEM v3\.[0-9]*\.[0-9]*" | sort -u | tr "\n" " ")
    echo "  $lib -> ${ver:-unknown}"
    case "$ver" in
        *"NVSHMEM v3.7.0"*) ;;
        *) echo "  FAIL: expected NVSHMEM v3.7.0"; fail=1 ;;
    esac
done

echo "--- NCCL aws-ofi-nccl plugin is loadable under the name NCCL will use ---"
# The EFA installer ships libnccl-net-ofi.so, not the libnccl-net.so that NCCL
# auto-loads, so NCCL_NET_PLUGIN=ofi must be set AND the .so must resolve through
# ldconfig. Getting this wrong is a silent fall back to TCP sockets.
if [ "${NCCL_NET_PLUGIN:-}" != "ofi" ]; then
    echo "FAIL: NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-<unset>}, expected \"ofi\" (see Dockerfile)"; fail=1
else
    python3 - <<PY || fail=1
import ctypes, sys
so = "libnccl-net-%s.so" % "ofi"   # exactly how NCCL templates the short name
try:
    ctypes.CDLL(so, ctypes.RTLD_GLOBAL)
except OSError as e:
    print("FAIL: NCCL would not find %s -> silent TCP fallback (%s)" % (so, e)); sys.exit(1)
print("OK:", so, "resolves via ldconfig")
PY
fi

echo "--- mooncake imports (EFA build) ---"
python3 -c "import mooncake; print(\"OK: mooncake\", mooncake.__file__)" || fail=1

echo "--- torch lib dir is on the loader path ---"
TORCH_LIB=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__) + \"/lib\")")
if [ -d "$TORCH_LIB" ]; then echo "OK: $TORCH_LIB"; else echo "FAIL: $TORCH_LIB missing"; fail=1; fi

echo
if [ "$fail" -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "SOME CHECKS FAILED"; exit 1; fi
'
