#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# run-kernel-test.sh — prove DeepEP-V2 dispatch/combine actually moves bytes over EFA on the
# GDAKI (GPU-initiated) transport BEFORE the ~ multi-hundred-GB model load. Runs DeepEP-V2's own
# elastic EP test (tests/elastic/test_ep.py) across the nodes with the SAME GDAKI-Gin/EFA env the
# serve uses, and exits non-zero on failure. Cheap, and the one step that cannot hang for hours.
#
#   leader:  run-kernel-test.sh leader <leader-ip>
#   worker:  run-kernel-test.sh worker <leader-ip> <node-rank>   # node-rank 1,2,3 for the worker nodes
#
# Env (shared with serve.sh): NNODES (default 2), GPUS_PER_NODE (default 8),
# OFI_NCCL_GDAKI_EFA_HW_COUNTER (default "off" here — set "auto" on efa.ko>=3.3.0 nodes).
set -uo pipefail

ROLE="${1:?usage: run-kernel-test.sh {leader|worker} <leader-ip> [node-rank]}"
LEADER_IP="${2:?need leader ip}"
NODE_RANK_ARG="${3:-0}"
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TEST="/opt/DeepEP/tests/elastic/test_ep.py"
[ -f "$TEST" ] || { echo "FAIL: $TEST not in image — rebuild from setup_deepep_v2_gdaki_efa.sh"; exit 3; }

# ---- GDAKI-Gin + EFA contract, VERBATIM from serve.sh (same transport under test) ----
export NCCL_GIN_TYPE=3 NCCL_GIN_ENABLE=1 OFI_NCCL_GIN_GDAKI=1 OFI_NCCL_GIN_MAX_REQUESTS=512
export OFI_NCCL_GDAKI_EFA_HW_COUNTER="${OFI_NCCL_GDAKI_EFA_HW_COUNTER:-off}"   # STRING enum; "off" on efa.ko<3.3.0
export NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IGNORE_DISABLED_P2P=1
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_ENABLE_SHM_TRANSFER=0 FI_EFA_FORK_SAFE=1
export OFI_NCCL_PROTOCOL=RDMA DEEP_EP_BACKEND=nccl
export NCCL_NET_PLUGIN=/opt/aws-ofi-nccl-gdaki/lib/libnccl-net-ofi.so
export OFI_NCCL_GDRCOPY_FORCED_PCIE_COPY=1
export EP_REUSE_NCCL_COMM=0          # DeepEP own-comm (segfault rootcause)
export NCCL_DEBUG=${KERNEL_TEST_NCCL_DEBUG:-INFO}   # INFO so the efa-direct banner prints = transport proof

NODE_RANK="$NODE_RANK_ARG"; [ "$ROLE" = "leader" ] && NODE_RANK=0
echo "===== DeepEP-V2 GDAKI kernel smoke: role=$ROLE node_rank=$NODE_RANK nnodes=$NNODES gpus/node=$GPUS_PER_NODE leader=$LEADER_IP hw_counter=$OFI_NCCL_GDAKI_EFA_HW_COUNTER $(hostname) $(date -u +%FT%TZ) ====="

set -o pipefail
torchrun --nnodes="$NNODES" --nproc-per-node="$GPUS_PER_NODE" --node-rank="$NODE_RANK" \
  --master-addr="$LEADER_IP" --master-port=29501 "$TEST" 2>&1 | tee /tmp/kernel-test.$NODE_RANK.log
rc=${PIPESTATUS[0]}

# watch for the known GDAKI failure signature (status-9 CQE) — a KNOWN upstream mode, not new
if grep -qiE "GDAKI-CQE .*status 9" /tmp/kernel-test.$NODE_RANK.log; then
  echo "KERNEL-TEST WARN: observed 'GDAKI-CQE ... status 9' — known upstream signature; capture the full log and correlate"
fi

# fail-loud contract: torchrun rc must be 0 AND the test must have printed a pass marker
if [ "$rc" -eq 0 ] && grep -qiE "passed|\bPASS\b|all tests? ok" /tmp/kernel-test.$NODE_RANK.log; then
  # confirm EFA (not TCP/SHM) actually carried it — the efa-direct/OFI banner from NCCL_DEBUG=INFO
  if grep -qiE "efa-direct|Selected provider is efa|NET/OFI" /tmp/kernel-test.$NODE_RANK.log; then
    echo "KERNEL-TEST PASS (node_rank=$NODE_RANK) — DeepEP-V2 dispatch/combine over EFA (GDAKI) verified"
    exit 0
  fi
  echo "KERNEL-TEST INCONCLUSIVE: test passed but no EFA banner in log — confirm transport before trusting"
  exit 2
fi
echo "KERNEL-TEST FAIL (node_rank=$NODE_RANK, torchrun rc=$rc) — see /tmp/kernel-test.$NODE_RANK.log"
exit 1
