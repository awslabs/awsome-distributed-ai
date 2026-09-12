#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# train-step.sh — the N-step Megatron-core MoE training gate over EFA (the training
# workload's analogue of a served smoke test). Runs recipe/train_moe_step.py under
# torchrun across the nodes with the SAME NCCL-GIN/EFA env the full GRPO run uses,
# and exits non-zero on any failure.
#
#   leader:  train-step.sh leader <leader-ip>
#   worker:  train-step.sh worker <leader-ip> <node-rank>   # node-rank 1,2,... for worker nodes
#
# Env: NNODES (default 2), GPUS_PER_NODE (default 8), MOE_DISPATCHER (default alltoall),
# TRAIN_STEPS (default 3), EP_EXPERTS/EP_TOPK/EP_HIDDEN (see env_vars.example).
set -uo pipefail

ROLE="${1:?usage: train-step.sh leader|worker <leader-ip> [node-rank]}"
case "$ROLE" in leader|worker) ;; *) echo "FATAL: unrecognized role '$ROLE' (leader|worker)"; exit 2 ;; esac
LEADER_IP="${2:?need leader ip}"
if [ "$ROLE" = "worker" ]; then NODE_RANK_ARG="${3:?worker requires an explicit node-rank (1,2,...)}"; else NODE_RANK_ARG=0; fi
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
DRIVER="/opt/train_moe_step.py"
[ -f "$DRIVER" ] || { echo "FAIL: $DRIVER not in image — rebuild from nemo-rl.Dockerfile"; exit 3; }

# ---- dispatcher selection guard ----
# flex (+moe_enable_deepep) is the DeepEP V2 ElasticBuffer path and needs the opt-in
# draft-PR image (Megatron-LM#4632 et al). On an unpatched image, refuse up front with
# the reason, rather than dying later inside megatron with an import error — and never
# silently fall back to a different dispatcher than the one requested.
export MOE_DISPATCHER="${MOE_DISPATCHER:-alltoall}"
if [ "$MOE_DISPATCHER" = "flex" ] && [ ! -f /opt/.draft-rollout-patches-applied ]; then
  echo "FATAL: MOE_DISPATCHER=flex (DeepEP V2 ElasticBuffer) needs an image built with"
  echo "APPLY_DRAFT_ROLLOUT_PATCHES=1 (bakes Megatron-LM#4632 + NeMo-RL#2410)."
  echo "This baseline image is upstream-only — run the default alltoall dispatcher gate,"
  echo "or rebuild with the opt-in layer."
  exit 4
fi

# ---- NCCL-GIN proxy + EFA contract, VERBATIM from run-rollout-probe.sh ----
export NCCL_GIN_TYPE=2 NCCL_GIN_ENABLE=1 OFI_NCCL_GIN_GDAKI=0
export NCCL_NVLS_ENABLE=0
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_ENABLE_SHM_TRANSFER=0 FI_EFA_FORK_SAFE=1
export OFI_NCCL_PROTOCOL=RDMA
export NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so
export NCCL_DEBUG="${TRAIN_NCCL_DEBUG:-INFO}"   # INFO so the efa provider banner prints = transport proof

NODE_RANK="$NODE_RANK_ARG"; [ "$ROLE" = "leader" ] && NODE_RANK=0
echo "===== Megatron MoE train-step: role=$ROLE node_rank=$NODE_RANK nnodes=$NNODES gpus/node=$GPUS_PER_NODE dispatcher=$MOE_DISPATCHER leader=$LEADER_IP $(hostname) $(date -u +%FT%TZ) ====="

# EFA hardware TX bytes across all NICs — same honest transport assert the rollout probe
# uses: on a multi-node run the MoE dispatcher's cross-node all-to-all (NCCL all-to-all for
# `alltoall`, DeepEP ElasticBuffer for `flex`) MUST move bytes over EFA, so a PASS with the
# counter flat means the traffic fell back to SHM/TCP. Sum the byte-counter FAMILIES, not
# tx_bytes alone: with FI_EFA_USE_DEVICE_RDMA=1 the bytes can be accounted as
# rdma_write_bytes rather than sends, so tx_bytes alone could read ~0 on a healthy RDMA run
# and false-FAIL. Missing families are skipped (the glob finds fewer files).
efa_tx_total() {
  local total=0 v
  for f in /sys/class/infiniband/*/ports/1/hw_counters/{tx_bytes,send_bytes,rdma_write_bytes,rdma_read_resp_bytes}; do
    [ -f "$f" ] && v=$(cat "$f") && total=$((total + v))
  done
  echo "$total"
}
TX_BEFORE=$(efa_tx_total)

set -o pipefail
# Wall-clock bound (timeout 900) so a wedged rendezvous returns a verdict instead of holding
# the nodes; train_moe_step.py also passes an init_process_group timeout. timeout exit 124 =>
# treated as a gate failure by the rc check below.
timeout 900 torchrun --nnodes="$NNODES" --nproc-per-node="$GPUS_PER_NODE" --node-rank="$NODE_RANK" \
  --master-addr="$LEADER_IP" --master-port=29502 "$DRIVER" 2>&1 | tee /tmp/train-step.$NODE_RANK.log
rc=${PIPESTATUS[0]}

TX_AFTER=$(efa_tx_total)
TX_DELTA=$((TX_AFTER - TX_BEFORE))
echo "EFA hw_counters tx_bytes delta on this node: $TX_DELTA"

# fail-loud contract, same shape as run-rollout-probe.sh: gate on the exit code (the driver
# all-reduces a MIN verdict — one bad rank fails every rank), then separately require EFA
# evidence, then confirm the provider banner.
if [ "$rc" -ne 0 ]; then
  echo "TRAIN-STEP GATE FAIL (node_rank=$NODE_RANK, torchrun rc=$rc) — see /tmp/train-step.$NODE_RANK.log"
  exit 1
fi
if [ "$NNODES" -gt 1 ] && [ "$TX_DELTA" -lt 1048576 ]; then
  # a real cross-node MoE all-to-all moves MBs; <1 MiB means the bytes went over SHM/TCP, not EFA
  echo "TRAIN-STEP GATE FAIL: step passed but EFA TX advanced only ${TX_DELTA}B — transport was NOT EFA"
  exit 1
fi
# only provider-specific banners count; a bare "NET/OFI" line also prints for the
# tcp;ofi_rxm fallback, the exact case to rule out
if grep -qiE "efa-direct|Selected Provider is efa" /tmp/train-step.$NODE_RANK.log; then
  echo "TRAIN-STEP GATE PASS (node_rank=$NODE_RANK, dispatcher=$MOE_DISPATCHER) — MoE step over EFA verified (tx +${TX_DELTA}B)"
  exit 0
fi
echo "TRAIN-STEP GATE INCONCLUSIVE: step passed and counters moved (+${TX_DELTA}B) but no EFA-provider banner in log — confirm transport before trusting"
exit 2
