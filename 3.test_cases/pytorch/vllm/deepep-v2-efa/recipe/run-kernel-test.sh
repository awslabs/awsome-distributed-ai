#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# run-kernel-test.sh — prove DeepEP-V2 dispatch/combine actually moves bytes over EFA BEFORE the
# ~ multi-hundred-GB model load. This is the "smoke-test the transport" gate: it runs DeepEP-V2's own
# elastic EP test (tests/elastic/test_ep.py) across the nodes with the SAME proxy-Gin/EFA env the
# serve uses, and exits non-zero on failure. Cheap, and the one step that cannot hang for hours.
#
#   leader:  run-kernel-test.sh leader <leader-ip>
#   worker:  run-kernel-test.sh worker <leader-ip> <node-rank>   # node-rank 1,2,3 for the worker nodes
#
# Env (shared with serve.sh): NNODES (default 2), GPUS_PER_NODE (default 8).
set -uo pipefail

ROLE="${1:?usage: run-kernel-test.sh {leader|worker} <leader-ip> [node-rank]}"
case "$ROLE" in leader|worker) ;; *) echo "FATAL: unrecognized role '$ROLE' (leader|worker)"; exit 2 ;; esac
LEADER_IP="${2:?need leader ip}"
if [ "$ROLE" = "worker" ]; then NODE_RANK_ARG="${3:?worker requires an explicit node-rank (1,2,...)}"; else NODE_RANK_ARG=0; fi
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TEST="/opt/DeepEP/tests/elastic/test_ep.py"
[ -f "$TEST" ] || { echo "FAIL: $TEST not in image — rebuild from setup_deepep_v2_efa.sh"; exit 3; }

# ---- proxy-Gin + EFA contract, VERBATIM from serve.sh (same transport under test) ----
export NCCL_GIN_TYPE=2 NCCL_GIN_ENABLE=1 OFI_NCCL_GIN_GDAKI=0 OFI_NCCL_GIN_MAX_REQUESTS=512
export NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IGNORE_DISABLED_P2P=1
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_ENABLE_SHM_TRANSFER=0 FI_EFA_FORK_SAFE=1
export OFI_NCCL_PROTOCOL=RDMA DEEP_EP_BACKEND=nccl
export NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so
export OFI_NCCL_GDRCOPY_FORCED_PCIE_COPY=1
export EP_REUSE_NCCL_COMM=0          # DeepEP own-comm (segfault rootcause 2026-08-14)
export NCCL_DEBUG=${KERNEL_TEST_NCCL_DEBUG:-INFO}   # INFO so the efa-direct banner prints = transport proof

NODE_RANK="$NODE_RANK_ARG"; [ "$ROLE" = "leader" ] && NODE_RANK=0
echo "===== DeepEP-V2 kernel smoke: role=$ROLE node_rank=$NODE_RANK nnodes=$NNODES gpus/node=$GPUS_PER_NODE leader=$LEADER_IP $(hostname) $(date -u +%FT%TZ) ====="

set -o pipefail
# ONE torchrun process per node: test_ep.py spawns its own local ranks internally
# (torch.multiprocessing.spawn with --num-processes, default 8) and DeepEP's init_dist
# reads WORLD_SIZE as a NODE count (num_nodes = WORLD_SIZE, world = nodes x local_ranks).
# --nproc-per-node=8 double-fans-out (2 nodes -> 16 procs x 8 spawns = 128 ranks on 16
# GPUs) and collides MASTER_PORT 29501 with torchrun's own rendezvous -> NCCL "invalid
# usage" at init_process_group, every time. One proc/node lets the test own the fan-out.
torchrun --nnodes="$NNODES" --nproc-per-node=1 --node-rank="$NODE_RANK" \
  --master-addr="$LEADER_IP" --master-port=29501 "$TEST" --num-processes "$GPUS_PER_NODE" 2>&1 | tee /tmp/kernel-test.$NODE_RANK.log
rc=${PIPESTATUS[0]}

# fail-loud contract: gate on the exit code (the test is assertion-based and prints no
# pass marker — any grep for one would fail a genuinely green run), then separately
# require the EFA transport banner.
if [ "$rc" -eq 0 ]; then
  # confirm EFA (not TCP/SHM) actually carried it — only provider-specific banners count;
  # a bare "NET/OFI" line also prints for tcp;ofi_rxm fallback, the exact case to rule out
  if grep -qiE "efa-direct|Selected Provider is efa" /tmp/kernel-test.$NODE_RANK.log; then
    echo "KERNEL-TEST PASS (node_rank=$NODE_RANK) — DeepEP-V2 dispatch/combine over EFA verified"
    exit 0
  fi
  echo "KERNEL-TEST INCONCLUSIVE: test passed but no EFA-provider banner in log — confirm transport before trusting"
  exit 2
fi
echo "KERNEL-TEST FAIL (node_rank=$NODE_RANK, torchrun rc=$rc) — see /tmp/kernel-test.$NODE_RANK.log"
exit 1
