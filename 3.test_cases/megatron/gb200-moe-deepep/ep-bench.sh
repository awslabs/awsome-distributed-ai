#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# DeepEP dispatch/combine bandwidth on GB200. Measures the crossover between:
#   SCOPE=intra : EP within one 72-GPU NVLink domain (NVLS, no EFA)
#   SCOPE=cross : EP across two UltraServers (all-to-all over EFA, no in-network SHARP)
# Asserts via NCCL/NVSHMEM logs that intra-domain EP issues no EFA traffic.
set -euo pipefail
: "${SCOPE:=intra}"

# Surface which transport DeepEP/NVSHMEM selects so the EFA-vs-NVLink claim is verifiable.
export NCCL_DEBUG=INFO
export NVSHMEM_DEBUG=INFO
export NVSHMEM_DEBUG_SUBSYS=TRANSPORT

source /opt/megatron-gb200/gb200-env.sh 2>/dev/null || true

case "$SCOPE" in
  intra) echo "[intra] EP within one 72-GPU NVLink domain -- expect NVLink/NVLS, no EFA frames" ;;
  cross) echo "[cross] EP across two UltraServers -- expect EFA (libfabric / NCCL-GIN), no in-network SHARP" ;;
  *) echo "Unknown SCOPE=$SCOPE (intra|cross)"; exit 2 ;;
esac

# DeepEP ships a test harness (tests/test_internode.py / test_intranode.py / test_low_latency.py).
TEST=tests/test_intranode.py
[[ "$SCOPE" == cross ]] && TEST=tests/test_internode.py
python3 "$TEST" 2>&1 | tee ep-bench-${SCOPE}.log

echo "---- transport check ----"
if [[ "$SCOPE" == intra ]]; then
  if grep -qiE 'efa|libfabric' ep-bench-${SCOPE}.log; then
    echo "WARN: EFA frames seen on an intra-domain run -- EP may be leaving the NVLink domain."
  else
    echo "OK: no EFA traffic on the intra-domain run (EP stayed on NVLink)."
  fi
fi
