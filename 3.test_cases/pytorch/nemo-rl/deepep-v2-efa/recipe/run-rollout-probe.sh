#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# run-rollout-probe.sh — prove DeepEP V2's dispatch/combine actually moves bytes over EFA
# BEFORE any model load or GRPO launch. Drives the REAL deep_ep.ElasticBuffer cross-node
# with the SAME NCCL-GIN/EFA env the training uses (probe_rollout.py), asserts the EFA
# hardware TX counters advanced, and exits non-zero on any failure.
#
#   leader:  run-rollout-probe.sh leader <leader-ip>
#   worker:  run-rollout-probe.sh worker <leader-ip> <node-rank>   # node-rank 1,2,... for worker nodes
#
# Env (shared with train-step.sh): NNODES (default 2), GPUS_PER_NODE (default 8),
# EP_EXPERTS/EP_TOKENS/EP_HIDDEN/EP_TOPK/EP_NUM_SMS/EP_NUM_QPS (see env_vars.example).
set -uo pipefail

ROLE="${1:?usage: run-rollout-probe.sh {leader|worker} <leader-ip> [node-rank]}"
case "$ROLE" in leader|worker) ;; *) echo "FATAL: unrecognized role '$ROLE' (leader|worker)"; exit 2 ;; esac
LEADER_IP="${2:?need leader ip}"
if [ "$ROLE" = "worker" ]; then NODE_RANK_ARG="${3:?worker requires an explicit node-rank (1,2,...)}"; else NODE_RANK_ARG=0; fi
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
PROBE="/opt/probe_rollout.py"
[ -f "$PROBE" ] || { echo "FAIL: $PROBE not in image — rebuild from nemo-rl.Dockerfile"; exit 3; }

# ---- NCCL-GIN proxy + EFA contract, VERBATIM (baked in the image ENV; re-exported so an
#      ad-hoc shell that scrubbed its env still probes the transport under test) ----
export NCCL_GIN_TYPE=2 NCCL_GIN_ENABLE=1 OFI_NCCL_GIN_GDAKI=0 OFI_NCCL_GIN_MAX_REQUESTS=512
export NCCL_NVLS_ENABLE=0
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_ENABLE_SHM_TRANSFER=0 FI_EFA_FORK_SAFE=1
export OFI_NCCL_PROTOCOL=RDMA
export NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so
# ---- probe shape (defaults = the Qwen3-30B-A3B / Wave-28 shape twin) ----
export EP_EXPERTS="${EP_EXPERTS:-128}" EP_TOKENS="${EP_TOKENS:-128}"
export EP_HIDDEN="${EP_HIDDEN:-2048}" EP_TOPK="${EP_TOPK:-8}"
# Explicit SM/QP counts: upstream DeepEP's auto-sizers are EFA-blind at the pinned SHA
# (DeepEP#612 is the fix; opt-in layer). 2 QPs is the value the #612 evidence validated.
export EP_NUM_SMS="${EP_NUM_SMS:-8}" EP_NUM_QPS="${EP_NUM_QPS:-2}"
export NCCL_DEBUG="${PROBE_NCCL_DEBUG:-INFO}"   # INFO so the efa provider banner prints = transport proof

NODE_RANK="$NODE_RANK_ARG"; [ "$ROLE" = "leader" ] && NODE_RANK=0
echo "===== DeepEP-V2 rollout probe: role=$ROLE node_rank=$NODE_RANK nnodes=$NNODES gpus/node=$GPUS_PER_NODE leader=$LEADER_IP $(hostname) $(date -u +%FT%TZ) ====="

# EFA hardware TX bytes across all NICs — the counter delta is the honest "bytes actually
# left this node over EFA" assert (a PASS on SHM/TCP fallback would show delta≈0).
# Sum the byte-counter FAMILIES, not tx_bytes alone: with FI_EFA_USE_DEVICE_RDMA=1 the
# bytes can be accounted as rdma_write_bytes rather than sends, so tx_bytes alone could
# read ~0 on a perfectly healthy RDMA run and false-FAIL the >=1 MiB check below. Missing
# families are skipped (the glob simply finds fewer files), so this is portable across
# kernels that expose different subsets.
efa_tx_total() {
  local total=0 v
  for f in /sys/class/infiniband/*/ports/1/hw_counters/{tx_bytes,send_bytes,rdma_write_bytes,rdma_read_resp_bytes}; do
    [ -f "$f" ] && v=$(cat "$f") && total=$((total + v))
  done
  echo "$total"
}
TX_BEFORE=$(efa_tx_total)

set -o pipefail
# GPUS_PER_NODE torchrun procs per node (one per GPU): NNODES x GPUS_PER_NODE = the EP16
# shape the Wave-28 measured run used on 2 nodes.
# Wall-clock bound (timeout 900) so a wedged rendezvous returns a verdict instead of
# holding the nodes; the python side also passes an init_process_group timeout. timeout
# exit 124 => treated as a probe failure by the rc check below.
timeout 900 torchrun --nnodes="$NNODES" --nproc-per-node="$GPUS_PER_NODE" --node-rank="$NODE_RANK" \
  --master-addr="$LEADER_IP" --master-port=29501 "$PROBE" 2>&1 | tee /tmp/rollout-probe.$NODE_RANK.log
rc=${PIPESTATUS[0]}

TX_AFTER=$(efa_tx_total)
TX_DELTA=$((TX_AFTER - TX_BEFORE))
echo "EFA hw_counters tx_bytes delta on this node: $TX_DELTA"

# fail-loud contract: gate on the exit code (the probe all-reduces a MIN verdict — one bad
# rank fails every rank), then separately require EFA evidence.
if [ "$rc" -ne 0 ]; then
  echo "ROLLOUT-PROBE FAIL (node_rank=$NODE_RANK, torchrun rc=$rc) — see /tmp/rollout-probe.$NODE_RANK.log"
  exit 1
fi
if [ "$NNODES" -gt 1 ] && [ "$TX_DELTA" -lt 1048576 ]; then
  # a real cross-node dispatch moves MBs; <1 MiB means the bytes went over SHM/TCP, not EFA
  echo "ROLLOUT-PROBE FAIL: probe passed but EFA TX advanced only ${TX_DELTA}B — transport was NOT EFA"
  exit 1
fi
# only provider-specific banners count; a bare "NET/OFI" line also prints for the
# tcp;ofi_rxm fallback, the exact case to rule out
if grep -qiE "efa-direct|Selected Provider is efa" /tmp/rollout-probe.$NODE_RANK.log; then
  echo "ROLLOUT-PROBE PASS (node_rank=$NODE_RANK) — ElasticBuffer dispatch/combine over EFA verified (tx +${TX_DELTA}B)"
  exit 0
fi
echo "ROLLOUT-PROBE INCONCLUSIVE: probe passed and counters moved (+${TX_DELTA}B) but no EFA-provider banner in log — confirm transport before trusting"
exit 2
